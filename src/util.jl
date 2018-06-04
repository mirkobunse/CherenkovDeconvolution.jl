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
using StatsBase, Discretizers, Polynomials, DataFrames

export fit_pdf, fit_R, edges, normalizetransfer
export normalizepdf, normalizepdf!, chi2s


"""    
    fit_pdf(x[, bins])

Obtain the discrete pdf of the integer array `x`, optionally specifying the array of `bins`.
"""
function fit_pdf{T<:Int}(x::AbstractArray{T,1}, bins::AbstractArray{T,1}=unique(x);
                         normalize::Bool = true)
    h = fit(Histogram, x, edges(bins), closed=:left).weights
    return normalize ? normalizepdf(h) : h
end


"""
    fit_R(y, x; bins_y, bins_x, normalize=true)

Estimate the detector response matrix `R`, which empirically captures the transfer from the
integer array `y` to the integer array `x`.

`R` is normalized by default so that `fit_pdf(x) == R * fit_pdf(y)`.
If `R` is not normalized now, you can do so later calling `Util.normalizetransfer(R)`.
"""
function fit_R{T<:Int}(y::AbstractArray{T,1}, x::AbstractArray{T,1};
                       bins_y::AbstractArray{T,1}=unique(y),
                       bins_x::AbstractArray{T,1}=unique(x),
                       normalize::Bool = true)
    # check arguments (not done by fit(::Histogram, ..))
    if length(y) != length(x)
        throw(ArgumentError("x and y have different dimensions"))
    end
    
    # estimate detector response matrix
    R = fit(Histogram, (x, y), (edges(bins_x), edges(bins_y)), closed=:left).weights
    return normalize ? normalizetransfer(R) : R
end


"""
    edges(x)

Obtain the edges of an histogram of the integer array `x`.
"""
function edges{T<:Int}(x::AbstractArray{T,1})
    xmin, xmax = extrema(x)
    return xmin:(xmax+1)
end


"""
    normalizetransfer(R)

Normalize each column in `R` to make a probability density function.
"""
function normalizetransfer{T<:Number}(R::AbstractMatrix{T})
    R_norm = zeros(Float64, size(R))
    for i in 1:size(R, 2)
        R_norm[:,i] = normalizepdf(R[:,i])
    end
    return R_norm
end


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


"""
    df2Xy(df, y, features = setdiff(names(df), [y])))

Convert the DataFrame `df` to a tuple of the feature matrix `X` and the target column `y`.
"""
df2Xy(df::AbstractDataFrame, y::Symbol, features::Array{Symbol,1}=setdiff(names(df), [y])) =
    df2X(df, features), convert(Array, df[y])

"""
    df2X(df, features = names(df))

Convert the DataFrame `df` to a feature matrix `X`.
"""
df2X(df::AbstractDataFrame, features::AbstractArray{Symbol,1}=names(df)) = convert(Matrix, df[:, features])

