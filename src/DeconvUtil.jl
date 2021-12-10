# 
# CherenkovDeconvolution.jl
# Copyright 2018-2021 Mirko Bunse
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
module DeconvUtil

using Discretizers, LinearAlgebra, Polynomials, ScikitLearn, StatsBase

export
    chi2s,
    cov_g,
    cov_multinomial,
    cov_Poisson,
    edges,
    expansion_discretizer,
    fit_pdf,
    fit_R,
    inspect_expansion,
    inspect_reduction,
    normalizepdf,
    normalizepdf!,
    normalizetransfer,
    polynomial_smoothing,
    reduce,
    train_and_predict_proba

"""    
    fit_pdf(x[, bins]; normalize=true, laplace=false)

Obtain the discrete pdf of the integer array `x`, optionally specifying the array of `bins`.

The result is normalized by default. If it is not normalized now, you can do so later by
calling `DeconvUtil.normalizepdf`.

Laplace correction means that at least one example is assumed in every bin, so that no bin
has probability zero. This feature is disabled by default.
"""
function fit_pdf(x::AbstractVector{T}, bins::AbstractVector{T}=unique(x);
                 normalize::Bool=true, laplace::Bool=false) where T<:Int
    h = StatsBase.fit(Histogram, x, edges(bins), closed=:left).weights
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
If `R` is not normalized now, you can do so later calling `DeconvUtil.normalizetransfer(R)`.
"""
function fit_R(y::AbstractVector{T}, x::AbstractVector{T};
               bins_y::AbstractVector{T}=unique(y),
               bins_x::AbstractVector{T}=unique(x),
               normalize::Bool=true) where T<:Int
    # check arguments (not done by fit(::Histogram, ..))
    if length(y) != length(x)
        throw(ArgumentError("x and y have different dimensions"))
    end
    
    # estimate detector response matrix
    R = StatsBase.fit(Histogram, (convert(Array, x), convert(Array, y)), (edges(bins_x), edges(bins_y)), closed=:left).weights
    return normalize ? normalizetransfer(R) : R
end

"""
    edges(x)

Obtain the edges of an histogram of the integer array `x`.
"""
function edges(x::AbstractVector{T}) where T<:Int
    xmin, xmax = extrema(x)
    return xmin:(xmax+1)
end

"""
    normalizetransfer(R[; warn=true])

Normalize each column in `R` to make a probability density function.
"""
function normalizetransfer(R::AbstractMatrix{T}; warn::Bool=true) where T<:Number
    R_norm = zeros(Float64, size(R))
    for i in 1:size(R, 2)
        R_norm[:,i] = normalizepdf(R[:,i], warn=warn)
    end
    return R_norm
end

_DOC_NORMALIZEPDF = """
    normalizepdf(array...; warn=true)
    normalizepdf!(array...; warn=true)

Normalize each array to a discrete probability density function.

By default, `warn` if coping with NaNs, Infs, or negative values.
"""
@doc _DOC_NORMALIZEPDF normalizepdf  # map doc string to function
@doc _DOC_NORMALIZEPDF normalizepdf!

normalizepdf(a::AbstractVector...; kwargs...) =
    normalizepdf!(map(ai -> map(Float64, ai), a)...; kwargs...)

function normalizepdf!(a::AbstractVector...; warn::Bool=true)
    arrs = [ a... ] # convert tuple to array
    single = length(a) == 1 # normalization of single array?
    
    # check for NaNs and Infs
    nans = [ any(isnan.(arr)) || any(abs.(arr) .== Inf) for arr in arrs ]
    if sum(nans) > 0
        for arr in map(i -> arrs[i], findall(nans))
            arr[isnan.(arr)] .= 0
            arr[abs.(arr) .== Inf] .= 0
            arr[:] .= abs.(arr) # float arrays can have negative zeros leading to more warnings
        end
        if warn && single
            @warn "Normalization set NaNs and Infs to zero"
        elseif warn
            @warn "Normalization set NaNs and Infs in $(sum(nans)) arrays to zero"
        end
    end
    
    # check for negative values
    negs = [ any(arr .< 0) for arr in arrs ]
    if sum(negs) > 0
        for arr in map(i -> arrs[i], findall(negs))
            arr[arr .< 0] .= 0
            arr[:] .= abs.(arr)
        end
        if warn && single
            @warn "Normalization set negative values to zero"
        elseif warn
            @warn "Normalization set negative values in $(sum(negs)) arrays to zero"
        end
    end
    
    # check for zero sums
    sums = map(sum, arrs)
    zers = map(iszero, sums)
    if sum(zers) > 0
        for arr in map(i -> arrs[i], findall(zers))
            idim = length(arr)
            arr[:] .= ones(idim) ./ idim
        end
        if warn && single
            @warn "Normalization replaced a zero vector by a uniform density"
        elseif warn
            @warn "Normalization replaced $(sum(zers)) zero vectors by uniform densities"
        end
    end
    
    # normalize (narrs array is created before assignment for cases where the same array is in arrs multiple times)
    narrs = [ zers[i] ? arrs[i] : arrs[i] ./ sums[i] for i in 1:length(arrs) ]
    for i in findall(.!(zers))
        arrs[i][:] .= narrs[i]
    end
    
    # return tuple or single value
    if single
        return arrs[1]
    else
        return (arrs...,) # convert to tuple
    end
end

_DOC_COV = """
    cov_Poisson(g, N)
    cov_multinomial(g, N)
    
    cov_Poisson(g)
    cov_multinomial(g)
    
    cov_g(g, N = sum(g), assumption = :Poisson)

Estimate the variance-covariance matrix of the bins with an observed `g`.

In the first form, `g` is a density and `N` is the total number of observations.
In the second form, `g` contains absolute counts, so that `N = sum(g)`. The third
form is a general shorthand for any of the above methods.

The method `cov_Poisson` assumes a Poisson distribution in each of the bins.
`cov_multinomial`, assumes a multinomial distribution over all bins.
"""
@doc _DOC_COV cov_Poisson
@doc _DOC_COV cov_multinomial
@doc _DOC_COV cov_g

cov_Poisson(g::Vector{T}, N::Integer) where T<:Real =
    cov_Poisson(round.(Int64, g.*N))

cov_Poisson(g::Vector{T}) where T<:Integer = diagm(0 => g) # Integer version, variance = mean

function cov_multinomial(g::Vector{T}, N::Integer) where T<:Real
    cov = zeros(length(g), length(g))
    for i in 1:size(cov, 1), j in 1:size(cov, 2)
        cov[i, j] = (i==j) ? N*g[i]*(1-g[i]) : -N*g[i]*g[j]
    end
    return cov
end

cov_multinomial(g::Vector{T}) where T<:Integer =
    cov_multinomial(normalizepdf(g), sum(g)) # Integer version

cov_g(g::Vector{T}, N::Integer = sum(Int64, g), assumption = :Poisson) where T<:Real =
    if assumption == :Poisson && eltype(g) <: Integer
        cov_Poisson(g) # counts version
    elseif assumption == :Poisson
        cov_Poisson(g, N) # density version
    elseif assumption == :multinomial && eltype(g) <: Integer
        cov_multinomial(g) # counts version
    elseif assumption == :multinomial
        cov_multinomial(g, N) # density version
    else
        throw(ArgumentError("assumption=$assumption must be either :Poisson or :multinomial"))
    end

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
            _repair_smoothing( Polynomials.fit(Float64.(1:length(f)), f, o).(1:length(f)), warn )
        else
            throw(ArgumentError("Impossible smoothing order $o >= dim(f) = $(length(f))"))
        end
    end

# post-process result of polynomial_smoothing (and other smoothing functions)
function _repair_smoothing(f::Vector{Float64}, warn::Bool)
    if any(f .< 0) # average values of neighbors for all values < 0
        for i in findall(f .< 0)
            f[i] = if i == 1
                       f[i+1] / 2
                   elseif i == length(f)
                       f[i-1] / 2
                   else
                       (f[i+1] + f[i-1]) / 4 # half the average [dagostini2010improved]
                   end
        end
        if warn # warn about negative values?
            Base.@warn "Smoothing averaged the values of neighbours to circumvent negative values"
        end
    end
    return normalizepdf(f, warn=warn)
end

"""
    expansion_discretizer(ld, factor)

Create a copy of the LinearDiscretizer `ld`, which uses a multiple of the original number of
bins. The resulting discretizer can be used to obtain an expanded problem from continuous
target values.
"""
function expansion_discretizer(ld::LinearDiscretizer, factor::Int)
    ymin, ymax = extrema(ld)
    num_bins = factor * length(bincenters(ld))
    return LinearDiscretizer(range(ymin, step = (ymax - ymin) / num_bins, length=num_bins + 1))
end

"""
    reduce(f, factor[, keepdim = false; normalize = true])

Reduce the deconvolution result `f`. The parameter `keepdim` specifies, if the result is
re-expanded to the original dimension of `f`, afterwards.

You can set `keepdim = false` to reduce solutions of a previously expanded deconvolution
problem. `keepdim = true` is useful if you reduce a solution of a non-expanded problem, and
want to compare the reduced result to non-reduced results.
"""
function reduce(f::Vector{T}, factor::Int, keepdim::Bool=false;
	            normalize::Bool=true) where T<:Number
    
    # combine bins of f
    imax   = length(f) - factor + 1 # maximum edge index
    iedges = 1:factor:imax          # indices of new edges with respect to f
    if length(f) % factor > 0       # add potentially missing edge
        iedges = vcat(iedges, imax + 1)
    end
    f_red = map(i -> sum(f[i:min(length(f), i + factor - 1)]), iedges)
    
    # keep input dimension, if desired
    if !keepdim
        return normalize ? normalizepdf(f_red, warn=false) : f_red
    else
        f_exp = vcat(map(v -> repeat([v], factor), f_red)...)[1:length(f)] # re-expand
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
function chi2s(a::AbstractVector{T}, b::AbstractVector{T}, normalize::Bool=true) where T<:Number
    if normalize
        a, b = normalizepdf(a, b, warn=false)
    end
    selection = .|(a .> 0, b .> 0) # limit computation to denominators > 0
    a = a[selection]
    b = b[selection]
    return 2 * sum((a .- b).^2 ./ (a .+ b)) # Distances.chisq_dist(a, b)
end

"""
    train_and_predict_proba(classifier, :sample_weight)

Obtain a `train_and_predict_proba` object for DSEA.

The optional argument gives the name of the `classifier` parameter with which the sample
weight can be specified when calling `ScikitLearn.fit!`. Usually, its value does not need to
be changed. However, if for example a scikit-learn `Pipeline` object is the `classifier`,
the name of the step has to be provided like `:stepname__sample_weight`.
"""
function train_and_predict_proba(classifier, sample_weight::Union{Symbol,Nothing}=:sample_weight)
    return (X_data::Any, X_train::Any, y_train::Vector, w_train::Vector) -> begin
        kwargs_fit = sample_weight == nothing ? [] : [ (sample_weight, DeconvUtil.normalizepdf(w_train)) ]
        ScikitLearn.fit!(classifier, X_train, y_train; kwargs_fit...)
        return ScikitLearn.predict_proba(classifier, X_data) # matrix of probabilities
    end
end

end # module
