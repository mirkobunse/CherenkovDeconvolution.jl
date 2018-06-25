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


using StatsBase, Discretizers, Polynomials, DataFrames

export fit_pdf, fit_R, edges, normalizetransfer
export normalizepdf, normalizepdf!, polynomial_smoothing, chi2s
export expansion_discretizer, reduce, inspect_expansion, inspect_reduction


"""    
    fit_pdf(x[, bins])

Obtain the discrete pdf of the integer array `x`, optionally specifying the array of `bins`.
"""
function fit_pdf{T<:Int}(x::AbstractArray{T,1}, bins::AbstractArray{T,1}=unique(x);
                         normalize::Bool=true, laplace::Bool=false)
    h = fit(Histogram, x, edges(bins), closed=:left).weights
    if laplace
        h = max.(h, 1)
    end
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
    R = fit(Histogram, (convert(Array, x), convert(Array, y)), (edges(bins_x), edges(bins_y)), closed=:left).weights
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
    polynomial_smoothing([o = 2, warn = true])

Create a function object `f -> smoothing(f)` which smoothes its argument with a polynomial
of order `o`. `warn` specifies if a warning is emitted when negative values returned by the
smoothing are replaced by the average of neighboring values - a post-processing step
proposed in [dagostini2010improved].
"""
polynomial_smoothing(o::Int=2, warn::Bool=true) =
    (f::Array{Float64,1}) -> begin # function object to be used as smoothing argument
        if o < length(f)
            # return the values of a fitted polynomial
            _repair_smoothing( polyval(polyfit(1:length(f), f, o), 1:length(f)), warn )
        else
            throw(ArgumentError("Impossible smoothing order $o >= dim(f) = $(length(f))"))
        end
    end

# post-process result of polynomial_smoothing (and other smoothing functions)
function _repair_smoothing(f::Array{Float64,1}, w::Bool)
    if any(f .< 0) # average values of neighbors for all values < 0
        if w # warn about negative values?
            warn("Averaging values of neighbours for negative values returned by smoothing")
        end
        for i in find(f .< 0)
            f[i] = if i == 1
                       f[i+1] / 2
                   elseif i == length(f)
                       f[i-1] / 2
                   else
                       (f[i+1] + f[i-1]) / 4 # half the average [dagostini2010improved]
                   end
        end
    end
    return normalizepdf(f)
end


"""
    expansion_discretizer(ld, factor)

Create a copy of the LinearDiscretizer `ld`, which uses a multiple of the original number of
bins. The resulting discretizer can be used to obtain an expanded problem from continuous
target values.
"""
function expansion_discretizer(ld::LinearDiscretizer, factor::Int)
    ymin, ymax = extrema(ld)
    num_bins   = length(bincenters(ld))
    return LinearDiscretizer(linspace(ymin, ymax, factor * num_bins + 1))
end

"""
    reduce(f, factor[, keepdim = false; normalize = true])

Reduce the deconvolution result `f`. The parameter `keepdim` specifies, if the result is
re-expanded to the original dimension of `f`, afterwards.

You can set `keepdim = false` to reduce solutions of a previously expanded deconvolution
problem. `keepdim = true` is useful if you reduce a solution of a non-expanded problem, and
want to compare the reduced result to non-reduced results.
"""
function reduce{TN<:Number}(f::Array{TN,1}, factor::Int, keepdim::Bool=false; normalize::Bool=true)
    
    # combine bins of f
    imax   = length(f) - factor + 1 # maximum edge index
    iedges = 1:factor:imax          # indices of new edges with respect to f
    if length(f) % factor > 0       # add potentially missing edge
        iedges = vcat(iedges, imax + 1)
    end
    f_red = map(i -> sum(f[i:min(length(f), i + factor - 1)]), iedges)
    
    # keep input dimension, if desired
    if !keepdim
        return normalize ? normalizepdf(f_red) : f_red
    else
        f_exp = vcat(map(v -> repmat([v], factor), f_red)...)[1:length(f)] # re-expand
        return normalize ? normalizepdf(f_exp) : f_exp
    end
    
end


"""
    inspect_expansion(inspect, factor)

Create a function object for the inspection of deconvolution methods, which wraps the given
`inspect` function so that expanded solutions are reduced for inspection. This helps you to
monitor the progress of a deconvolution method operating on an expanded problem.
"""
inspect_expansion(inspect::Function, factor::Int) =
    (f, args...) -> inspect(reduce(f, factor), args...)


"""
    inspect_reduction(inspect, factor)

Create a function object for the inspection of deconvolution methods, which wraps the given
`inspect` function so that reduced solutions are inspected.
"""
inspect_reduction(inspect::Function, factor::Int) =
    (f, args...) -> inspect(reduce(f, factor, true), args...)


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


end

