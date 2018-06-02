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

using DataFrames, Discretizers, Polynomials # TODO DataFrames needed?

export histogram, empiricaltransfer, normalizetransfer, normalizetransfer!
export normalizepdf, normalizepdf!, chi2s


"""    
    fit_pdf(x, edges)
    fit_pdf(x, ld)

Obtain the discrete pdf of the array `x` over the bins specified by `edges`. Alternatively,
obtain these edges from the `LinearDiscretizer` object `ld`.
"""
fit_pdf{T<:Number}(x::AbstractArray{T, 1}, ld::LinearDiscretizer) = fit_pdf(x, binedges(ld))

fit_pdf{T<:Number, N<:Real}(x::AbstractArray{T, 1}, edges::Vector{N}) =
    normalize(fit(Histogram, x, edges, closed=:left), mode=:probability).weights


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


# TODO replace histogram() by fit_pdf


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


# TODO replace empiricaltransfer() by fit_R, which uses StatsBase.histogram, too


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
    # TODO find more efficient implementation (like in CherenkovDeconvolution.py)
    
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


end
