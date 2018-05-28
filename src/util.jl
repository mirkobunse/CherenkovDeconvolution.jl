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
using MLDataUtils, YAML, Discretizers, Polynomials, Distances


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

If `d` is of type `Dict{Any,Any}` (as obtained from a YAML configuration), create
a discretization from that configuration; keyword arguments apply.
"""
function Discretization(d::Union{Discretization, Dict{Any,Any}}; kwargs...)
    argdict = Dict{Symbol,Any}(kwargs)
    if !isempty(setdiff(keys(argdict), [:name, :min, :max, :num_levels, :logscale]))
        throw(MethodError(Discretization(d; kwargs...)))
    end
    Discretization(_args_Discretization(d, argdict)...)
end

_args_Discretization(d::Dict{Any,Any}, argdict::Dict{Symbol,Any}) =
     Symbol(haskey(argdict, :name)     ? argdict[:name]     : d["name"]),
    Float64(haskey(argdict, :min)      ? argdict[:max]      : d["min"]),
    Float64(haskey(argdict, :max)      ? argdict[:max]      : d["max"]),
      Int64(haskey(argdict, :num_levels) ? argdict[:num_levels] : d["num_levels"]),
       Bool(haskey(argdict, :logscale) ? argdict[:logscale] : get(d, "logscale", false))

_args_Discretization(d::Discretization, argdict::Dict{Symbol,Any}) =
     Symbol(get(argdict, :name,     d.name)),
    Float64(get(argdict, :min,      d.min)),
    Float64(get(argdict, :max,      d.max)),
      Int64(get(argdict, :num_levels, d.num_levels)),
   Function(haskey(argdict, :logscale) ? _scaling(argdict[:logscale]) : d.scaling)

"""
    Discretization(configfile, key=""; kwargs...)

Read the discretization from a YAML configuration file, optionally changing values.
The keyword arguments are `name`, `min`, `max`, `num_levels` and `logscale`.
"""
function Discretization(configfile::AbstractString, key::AbstractString=""; kwargs...)
    c = YAML.load_file(configfile)
    Discretization(key != "" ? c[key] : c; kwargs...)
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
    histogram(arr; levels = nothing)

Return a histogram DataFrame with a `:level` column (unique values in `arr`) and an `:n`
column (number of respective occurences). The expected unique values in `arr` are optionally
supplied by the `level` argument.
"""
function histogram{T <: Number}(arr::AbstractArray{T,1};
                                levels::Union{AbstractArray{T,1},Void} = nothing)
    df = DataFrame(arr = arr, n = repmat([1], length(arr)))
    df = aggregate(df, :arr, sum) # compute histogram
    rename!(df, names(df), [:level, :n])
    if levels != nothing
        for missing in setdiff(levels, df[:level])
            push!(df, [missing, 0])
        end
        sort!(df, cols=[:level])
    end
    return df
end

"""
    histogram(df, col; levels = nothing)

Return a histogram of the column `col` in the DataFrame `df`.
"""
histogram(df::AbstractDataFrame, col::Symbol; kwargs...) = rename(histogram(df[col]; kwargs...), :n, col)



"""
    empiricaltransfer(df, x, y; normalize=true)

Empirically estimate the transfer matrix from the `y` column to the `x` column in the
DataFrame `df` where both columns are discrete, i.e., they have a limited number of unique
numerical values.

The returned matrix `R` is normalized by default so that `x_vec = R * y_vec` holds where
- `R = empiricaltransfer(df, x, y)`
- `x_vec = Util.histogram(df, x)[x]`
- `y_vec = Util.histogram(df, y)[y]`

If `R` is not normalized now, you can do so later calling `Util.normalizetransfer(R)`.
"""
function empiricaltransfer{T1 <: Number, T2 <: Number}(
            df::DataFrame, y::Symbol, x::Symbol;
            normalize::Bool = true,
            xlevels::AbstractArray{T1, 1} = sort(unique(df[x])),
            ylevels::AbstractArray{T2, 1} = sort(unique(df[y])))
    
    # count transfer occurences for each combination of levels
    counts = aggregate(hcat(df[:, [y, x]], DataFrame(c = ones(size(df, 1)))),
                       [y, x], sdf -> size(sdf, 1))
    
    # convert to matrix
    R = zeros(Float64, (length(xlevels), length(ylevels)))
    for k in 1:size(counts, 1)
        i = findfirst(ylevels, counts[k, y])
        j = findfirst(xlevels, counts[k, x])
        R[j,i] = counts[k,end]
    end
    
    if normalize
        return normalizetransfer!(R)
    else
        return R
    end
    
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
    normalizepdf!(df, columns=names(df))

Normalize the `columns` of the DataFrame `df` to discrete probability density functions.
"""
function normalizepdf!(df::DataFrame, columns::AbstractArray{Symbol,1}=names(df))
    for c in columns
        df[c] = normalizepdf(df[c])
    end
end

"""
    normalizepdf(array...)
    normalizepdf!(array...)

Normalize each array to a discrete probability density function.
"""
normalizepdf(a::AbstractArray...) = normalizepdf!(map(copy, a)...)

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
    subsample_uniformness([rng = GLOBAL_RNG,] df, y, u)

Subsample DataFrame `df` to attain the uniformness factor `u` in the `y` column.
Use `rng` to shuffle the subsample.

### Keyword arguments

- `levels = nothing` is an optional array of unique values in the `y` column
- `auxdf = nothing` is an auxiliary DataFrame to be used when `df` is small
- `shuffle = true` specifies, if shuffling should be performed
"""
subsample_uniformness(df::DataFrame, y::Symbol, u::Float64; kwargs...) =
    subsample_uniformness(Base.GLOBAL_RNG, df, y, u; kwargs...)

function subsample_uniformness(rng::AbstractRNG,
                               df::DataFrame, y::Symbol, u::Float64;
                               levels::AbstractArray{Float64, 1} = unique(df[y]),
                               auxdf::DataFrame = DataFrame(),
                               shuffle::Bool = true)
    
    # minimum bin size in original histogram = size of all bins in fully uniform subsample
    df_hist    = histogram(df, y, levels = levels)              # original histogram
    adf        = vcat(df, auxdf)                                # combined training set
    minbinsize = minimum(histogram(adf, y, levels = levels)[y]) # min bin size in adf
    
    # bin size after subsampling
    fbinsize = x -> minbinsize + (1 - u)*(x - minbinsize) # function
    binsizes = convert.(Int64, round.(map(fbinsize, df_hist[y]))) # mapping
    
    # subsample  data set
    adf = vcat([ adf[adf[y] .== binvalue, :][1:binsize, :]
                 for (binvalue, binsize) in zip(df_hist[:level], binsizes) ]...)
    return shuffle ? adf[randperm(rng, size(adf, 1)), :] : adf # shuffle
    
end

"""
    chi2s(a, b)

Symmetric Chi Square distance between histograms `a` and `b`.
"""
chi2s{T<:Number}(a::AbstractArray{T,1}, b::AbstractArray{T,1}) = 2 * Distances.chisq_dist(a, b)

