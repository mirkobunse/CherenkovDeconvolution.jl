# 
# CherenkovDeconvolution.jl
# Copyright 2018 Mirko Bunse
# 
# 
# Deconvolution methods for Cherenkov astronomy and other use cases in experimental physics.
# 
# 
# CherenkovDeconvolution.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CherenkovDeconvolution.jl.  If not, see <http://www.gnu.org/licenses/>.
# 
module Util

using DataFrames, Requires, Discretizers, Polynomials

export Discretization, levels, discretize, discretize!
export histogram, empiricaltransfer, normalizetransfer, normalizetransfer!
export normalizepdf, normalizepdf!, chi2s


"""
    Discretization(v, min, max, n, scaling = Base.identity)
    Discretization(v, min, max, n, logscale = false)

Linear discretization for the variable `v` that goes from `min` to `max` in `n` equidistant
bins. The resulting object is used in `Util.discretize()`.

The optional `scaling` transforms any value that is discretized (including `min` and `max`).
`scaling` is `Base.log10` if `logscale == true`.
"""
type Discretization
    name::Symbol
    min::Number
    max::Number
    num_levels::Int64
    edges::StepRangeLen{Float64}
    scaling::Function
    
    function Discretization(name::Symbol, min::Number, max::Number, num_levels::Int64, scaling::Function=identity)
        minval = scaling(min)
        maxval = scaling(max)
        new(name, min, max, num_levels,  minval:((maxval - minval) / num_levels):maxval,  scaling)
    end
end

Discretization(name::Symbol, min::Number, max::Number, num_levels::Int64, logscale::Bool=false) =
        Discretization(name, min, max, num_levels, _scaling(logscale))

_scaling(logscale::Bool) = logscale ? log10 : identity

"""
    Discretization(d; kwargs...)

Copy the `Discretization` object `d`, optionally changing values in the copy.
The keyword arguments are `name`, `min`, `max`, `num_levels` and `logscale`.
"""
function Discretization(d::Discretization; kwargs...)
    argdict = Dict{Symbol,Any}(kwargs)
    if !isempty(setdiff(keys(argdict), [:name, :min, :max, :num_levels, :logscale]))
        throw(MethodError(Discretization(d; kwargs...)))
    end
    Discretization(_args_Discretization(d, argdict)...)
end

_args_Discretization(d::Discretization, argdict::Dict{Symbol,Any}) =
     Symbol(get(argdict, :name,       d.name)),
    Float64(get(argdict, :min,        d.min)),
    Float64(get(argdict, :max,        d.max)),
      Int64(get(argdict, :num_levels, d.num_levels)),
   Function(haskey(argdict, :logscale) ? _scaling(argdict[:logscale]) : d.scaling)

@require YAML begin
    import YAML
    """
        Discretization(configfile, key=""; kwargs...)

    Read the discretization from a YAML configuration file, optionally changing values.
    The keyword arguments are `name`, `min`, `max`, `num_levels` and `logscale`.
    """
    function Discretization(configfile::AbstractString, key::AbstractString=""; kwargs...)
        c = YAML.load_file(configfile)
        d = Discretization(key != "" ? c[key] : c)
        isempty(kwargs) ? d : Discretization(d; kwargs...)
    end
end

"""
    levels(d::Discretization)

The unique values of data discretized with `d`.
"""
levels(d::Discretization) = collect(d.edges[1:end-1])

"""
    discretize(x, d::Discretization)
    discretize(df, d::Discretization...)

Return a discretized copy of the array `x` or the DataFrame `df`.
In a DataFrame, multiple columns can be discretized simultaneously.
"""
discretize(x::Union{Float64, AbstractArray{Float64,1}}, d::Discretization) =
        d.edges[encode(LinearDiscretizer(d.edges), d.scaling(x))]

discretize(df::AbstractDataFrame, ds::Discretization...) = discretize!(copy(df), ds...)

"""
    discretize!(df, d::Discretization...)

In-place variant of discretize(df, d::Discretization...)
"""
function discretize!(df::AbstractDataFrame, ds::Discretization...)
    for d in ds
        df[d.name] = discretize(df[d.name], d)
    end
    return df
end




"""
    histogram(arr, levels = nothing)

Return a histogram of `arr`, in which the unique values are optionally defined as `levels`.
"""
function histogram{T <: Number}(arr::AbstractArray{T,1}, levels::AbstractArray{T,1} = T[])
    
    # TODO replace by fit(Histogram, data[, weight][, edges]; closed=:right, nbins)
    # http://juliastats.github.io/StatsBase.jl/stable/empirical.html#StatsBase.fit-Tuple{Type{StatsBase.Histogram},Vararg{Any,N}%20where%20N}
    
    df = DataFrame(arr = arr, n = repmat([1], length(arr)))
    df = aggregate(df, :arr, sum) # compute histogram
    rename!(df, names(df), [:level, :n])
    if !isempty(levels)
        for missing in setdiff(levels, df[:level])
            push!(df, [missing, 0])
        end
        sort!(df, cols=[:level])
    end
    return convert(Array{Int64,1}, df[:n])
    
end


"""
    empiricaltransfer(y, x; normalize=true)

Empirically estimate the transfer matrix from the `y` array to the `x` array, both of which
are discrete, i.e., they have a limited number of unique values.

The returned matrix `R` is normalized by default so that `xhist = R * yhist` holds where
- `R = empiricaltransfer(y, x)`
- `xhist = histogram(x)`
- `yhist = histogram(y)`

If `R` is not normalized now, you can do so later calling `Util.normalizetransfer(R)`.
"""
function empiricaltransfer{T1 <: Number, T2 <: Number}(
            y::AbstractArray{T1, 1}, x::AbstractArray{T2, 1};
            normalize::Bool = true,
            ylevels::AbstractArray{T2, 1} = sort(unique(y)),
            xlevels::AbstractArray{T1, 1} = sort(unique(x)))
    
    # TODO test same length
    
    # count transfer occurences for each combination of levels
    counts = aggregate(DataFrame(y = y, x = x, c = ones(length(y))),
                       [:y, :x], sdf -> size(sdf, 1))
    
    # convert to matrix
    R = zeros(Float64, (length(xlevels), length(ylevels)))
    for k in 1:size(counts, 1)
        i = findfirst(ylevels, counts[k, :y])
        j = findfirst(xlevels, counts[k, :x])
        R[j,i] = counts[k,end]
    end
    
    if normalize
        normalizetransfer!(R)
    end
    return R
    
end

"""
    normalizetransfer(R)
    normalizetransfer!(R)

Normalize each column in `R` to make a probability density function.
"""
normalizetransfer(R::AbstractArray{Float64,2}) = normalizetransfer!(copy(R))

function normalizetransfer!(R::AbstractArray{Float64,2})
    for i in 1:size(R, 2)
        R[:,i] = normalizepdf(R[:,i])
    end
    return R
end

"""
    smoothpdf(arr, method=:polynomial; kwargs...)

Smooth the array values by a fit of the given order [dagostini2010improved].
"""
function smoothpdf{T<:Number}(abstractarr::AbstractArray{T,1}, method::Symbol=:polynomial; kwargs...)
    arr = convert(Array, abstractarr) # fixes bug occuring with DataArrays
    scale = sum(arr) # number of examples, or 1 for pdf input
    
    # actual smoothing
    arr = if method == :none
              copy(arr) # no smoothing, for convenience
          elseif method == :polynomial
              smooth_polynomial(arr; kwargs...)
          elseif method == :gmm
              smooth_gmm(arr; kwargs...)
          else
              error("Smoothing method '$method' is not valid")
          end
    
    # replace values < 0
    if any(arr .< 0)
        warn("Smoothing results in values < 0. Averaging values of neighbours for these.")
        for i in find(arr .< 0)
            arr[i] = if i == 1
                         arr[i+1] / 2
                     elseif i == length(arr)
                         arr[i-1] / 2
                     else
                         (arr[i+1] + arr[i-1]) / 4 # half the average [dagostini2010improved]
                     end
        end
    end
    
    # normalize and rescale
    (if sum(arr) > 0
        arr ./ sum(arr)
    else
        warn("Smoothing results in zero-array. Returning uniform distribution, instead.")
        repmat([ 1/length(arr) ], length(arr))
    end) .* scale
    
end

smooth_polynomial{T<:Number}(arr::Array{T,1}; order::Int=2) =
    if order < length(arr) # precondition
        polyval(polyfit(1:length(arr), arr, order), 1:length(arr)) # values of fitted polynomial
    else
        error("Polynomial smoothing with order = $order > length(arr) = $(length(arr)) is not possible")
    end

smooth_gmm{T<:Number}(arr::AbstractArray{T,1}; n::Int=1) =
    error("Not yet implemented") # values of fitted Gaussian Mixture Model

"""
    normalizepdf(array...)
    normalizepdf!(array...)

Normalize each array to a discrete probability density function.
"""
normalizepdf(a::AbstractArray...) = normalizepdf!(map(ai -> map(Float64, ai), a)...)

"""
    normalizepdf(array...)
    normalizepdf!(array...)

Normalize each array to a discrete probability density function.
"""
function normalizepdf!(a::AbstractArray...)
    
    arrs = [ a... ] # convert tuple to array
    single = length(a) == 1 # normalization of single array?
    
    # check for NaNs and Infs
    nans = [ any(isnan.(arr)) || any(abs.(arr) .== Inf) for arr in arrs ]
    if sum(nans) > 0
        if _WARN_NORMALIZE
            warn("Setting NaNs and Infs ",
                 single ? "" : "in $(sum(nans)) arrays ",
                 "to zero")
        end
        for arr in map(i -> arrs[i], find(nans))
            arr[isnan.(arr)] = 0
            arr[abs.(arr) .== Inf] = 0
            arr[:] = abs.(arr) # float arrays can have negative zeros leading to more warnings
        end
    end
    
    # check for negative values
    negs = [ any(arr .< 0) for arr in arrs ]
    if sum(negs) > 0
        if _WARN_NORMALIZE
            warn("Setting negative values ",
                 single ? "" : "in $(sum(negs)) arrays ",
                 "to zero")
        end
        for arr in map(i -> arrs[i], find(negs))
            arr[arr .< 0] = 0
            arr[:] = abs.(arr)
        end
    end
    
    # check for zero sums
    sums = map(sum, arrs)
    zers = map(iszero, sums)
    if sum(zers) > 0
        if _WARN_NORMALIZE
            warn(single ? "zero vector " : "$(sum(zers)) zero vectors ",
                 "replaced by uniform distribution",
                 single ? "" : "s")
        end
        for arr in map(i -> arrs[i], find(zers))
            idim   = length(arr)
            arr[:] = ones(idim) ./ idim
        end
    end
    
    # normalize (narrs array is created before assignment for cases where the same array is in arrs multiple times)
    narrs = [ zers[i] ? arrs[i] : arrs[i] ./ sums[i] for i in 1:length(arrs) ]
    for i in find(.!(zers))
        arrs[i][:] = narrs[i]
    end
    
    # return tuple or single value
    if single
        return arrs[1]
    else
        return (arrs...) # convert to tuple
    end
    
end
_WARN_NORMALIZE = true


"""
    chi2s(a, b, normalize = true)

Symmetric Chi Square distance between histograms `a` and `b`.
"""
function chi2s{T<:Number}(a::AbstractArray{T,1}, b::AbstractArray{T,1}, normalize = true)
    if normalize
        a, b = normalizepdf(a, b)
    end
    selection = .|(a .> 0, b .> 0) # limit computation to denominators > 0
    a = a[selection]
    b = b[selection]
    return 2 * sum((a .- b).^2 ./ (a .+ b)) # Distances.chisq_dist(a, b)
end


"""
    df2Xy(df, y, features = setdiff(names(df), [y])))

Convert the DataFrame `df` to a tuple of the feature matrix and the target column `y`.
"""
df2Xy(df::AbstractDataFrame, y::Symbol,
      features::AbstractArray{Symbol, 1} = setdiff(names(df), [y])) =
    convert(Array{Float64, 2}, df[:, features]),
    in(y, names(df)) ? convert(Array, df[y]) : nothing

"""
    prob2df(prob, ylevels) 

Convert the probability matrix `prob` to a DataFrame, where `ylevels` gives the column names.
"""
prob2df{T <: Any}(prob::AbstractArray{Float64, 2}, ylevels::AbstractArray{T, 1}) =
    DataFrame(; zip(map(Symbol, ylevels), [ prob[:,j] for j in 1:size(prob, 2) ])...)


end
