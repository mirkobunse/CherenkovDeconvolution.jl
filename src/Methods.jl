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
"""
    module Methods

This module contains a collection of deconvolution methods.
"""
module Methods

using LinearAlgebra, Optim
using ..DeconvUtil, ..Binnings, ..Stepsizes

export
    check_arguments,
    check_prior,
    decode_estimate,
    DeconvolutionMethod,
    deconvolve,
    DiscreteMethod,
    encode_labels,
    encode_prior,
    LabelSanitizer,
    LoneClassException,
    recover_estimate

export DSEA, IBU, PRUN, RUN, SVD # see src/methods/

"""
    abstract type DeconvolutionMethod

The supertype of all deconvolution methods.
"""
abstract type DeconvolutionMethod end

"""
    abstract type DiscreteMethod <: DeconvolutionMethod

The supertype of all classical deconvolution methods which estimate the density
function `f` from a transfer matrix `R` and an observed density `g`.
"""
abstract type DiscreteMethod <: DeconvolutionMethod end

"""
    expected_n_bins_y(y::AbstractVector{I<:Integer})
    expected_n_bins_y(m::DiscreteMethod)

Return the number of target bins that is to be expected from either a discrete
DeconvolutionMethod `m` or a vector of labels `y`.

The expectation of a method `m` is typically determined by it's configuration;
a pre-specified number of bins is to be expected.

The expectation of a label vector `y` ensures that missing numbers between `1`
and `maximum(y)` are assumed to be valid bins that just do not occur in `y`.
It further ensures that indices starting at `0` (as it is typical in Python)
will be interpreted as indices starting at `1` (as it is typical in Julia). This
behaviour is implemented as:

    expected_n_bins_y(y) = maximum(y) + (minimum(y) == 0)
"""
expected_n_bins_y(y::AbstractVector{I}) where {I<:Integer} = maximum(y) + (minimum(y) == 0)

"""
    LabelSanitizer(y_trn, n_bins=expected_n_bins_y(y_trn))

A sanitizer that

- encodes labels and priors so that none of the resulting bins is empty.
- decodes deconvolution results to recover the original (possibly empty) bins.

**See also:** `encode_labels`, `encode_prior`, `decode_estimate`.
"""
struct LabelSanitizer
    bins::Vector{Int} # bins that actually appear
    n_bins::Int # assumed number of bins
    LabelSanitizer(
            y_trn::AbstractVector{I},
            n_bins::Int=expected_n_bins_y(y_trn)
            ) where {I<:Integer} =
        new(sort(unique(y_trn)), n_bins)
end

"""
    deconvolve(m, X_obs, X_trn, y_trn)

Deconvolve the observed features in `X_obs` with the deconvolution method `m`
trained on the features `X_trn` and the corresponding labels `y_trn`.
"""
deconvolve(
        m::DeconvolutionMethod,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer} =
    throw(ArgumentError("Implementation missing for $(typeof(m))")) # must be implemented for sub-types

# discrete methods actually deconvolve from R and g, so the general API must wrap them
function deconvolve(
        m::DiscreteMethod,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer}

    # sanitize and check the arguments
    n_bins_y = max(expected_n_bins_y(m), expected_n_bins_y(y_trn)) # number of classes/bins
    try
        check_arguments(X_obs, X_trn, y_trn)
    catch exception
        if isa(exception, LoneClassException)
            f_est = recover_estimate(exception, n_bins_y)
            @warn "Only one label in the training set, returning a trivial estimate" f_est
            return f_est
        else
            rethrow()
        end
    end
    label_sanitizer = LabelSanitizer(y_trn, n_bins_y)
    y_trn = encode_labels(label_sanitizer, y_trn) # encode labels for safety
    initialize!(stepsize(m), X_obs, X_trn, y_trn) # initialize stepsizes

    # also check the optional prior
    f_0 = prior(m)
    if f_0 != nothing
        check_prior(f_0, n_bins_y)
        f_0 = DeconvUtil.normalizepdf(encode_prior(label_sanitizer, f_0))
    end

    # discretize the problem statement into a system of linear equations
    d = BinningDiscretizer(binning(m), X_trn, y_trn) # fit the binning strategy with labeled data
    x_obs = encode(d, X_obs) # apply it to the feature vectors
    x_trn = encode(d, X_trn)
    R = DeconvUtil.fit_R(y_trn, x_trn; bins_x=bins(d), normalize=expects_normalized_R(m))
    g = DeconvUtil.fit_pdf(x_obs, bins(d); normalize=expects_normalized_g(m))
    f_trn = DeconvUtil.fit_pdf(y_trn)

    # call the actual solver (IBU, RUN, etc)
    return deconvolve(m, R, g, label_sanitizer, f_trn, f_0)
end

# required API for discrete methods
deconvolve(
        m::DiscreteMethod,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f},
        f_0::Union{Vector{T_f},Nothing}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number} =
    throw(ArgumentError("Implementation missing for $(typeof(m))")) # must be implemented for sub-types
binning(m::DiscreteMethod) = throw(ArgumentError("Implementation missing for $(typeof(m))"))
stepsize(m::DiscreteMethod) = DEFAULT_STEPSIZE # default
prior(m::DiscreteMethod) = nothing
expects_normalized_R(m::DiscreteMethod) = false
expects_normalized_g(m::DiscreteMethod) = true
expected_n_bins_y(m::DiscreteMethod) = 0

# deconvolution methods
include("methods/dsea.jl")
include("methods/ibu.jl")
include("methods/prun.jl")
include("methods/run.jl")
include("methods/svd.jl")

"""
    LoneClassException(label)

An exception thrown by `check_arguments` when only one class is in the training set.

**See also:** `recover_estimate`
"""
struct LoneClassException <: Exception
    label::Int
end
Base.show(io::IO, x::LoneClassException) =
    print(io, "LoneClassException($(x.label)): Only a single label occurs in training set.")

"""
    recover_estimate(x::LoneClassException, n_bins=1)

Recover a trivial deconvolution result from `x`, in which all bins are zero, except
for the one that occured in the training set.
"""
function recover_estimate(x::LoneClassException, n_bins::Int=1)
    f_est = zeros(n_bins)
    f_est[x.label] = 1
    return f_est
end

"""
    check_arguments(X_obs, X_trn, y_trn)

Throw meaningful exceptions if the input data of a deconvolution run is defective.
"""
check_arguments(X_obs::Any, X_trn::Any, y_trn::AbstractVector{I}) where {I<:Integer} =
    if all(y_trn .== y_trn[1])
        throw(LoneClassException(y_trn[1]))
    end

"""
    check_discrete_arguments(R, g)

Throw meaningful exceptions if the input data of a discrete deconvolution is defective.
"""
check_discrete_arguments(R::Matrix{T_R}, g::AbstractArray{T_g}) where {T_R<:Number,T_g<:Number} =
    if size(R, 1) != length(g)
        throw(ArgumentError("dim(g) = $(length(g)) != $(size(R, 1)), the observable dimension of R"))
    end

"""
    check_prior(f_0, n_bins)

Throw meaningful exceptions if the input prior of a deconvolution run is defective.
"""
check_prior(f_0::AbstractVector{T}, n_bins::Int) where {T<:Number} =
    if length(f_0) != n_bins
        throw(ArgumentError("dim(f_0) = $(length(f_0)) != $(n_bins), the number of bins"))
    end

"""
    encode_labels(s::LabelSanitizer, y_trn)

Encode the labels `y_trn` so that all values from `1` to `max(y_trn)` occur.

**See also:** `encode_prior`, `decode_estimate`.
"""
function encode_labels(s::LabelSanitizer, y_trn::AbstractVector{I}) where {I<:Integer}
    encoder = Dict(zip(s.bins, 1:length(s.bins)))
    return map(y -> encoder[y], y_trn)
end

"""
    encode_prior(s::LabelSanitizer, f_0)

Encode the prior `f_0` to be consistent with the encoded labels.

**See also:** `encode_labels`, `decode_estimate`.
"""
function encode_prior(s::LabelSanitizer, f_0::AbstractVector{T}) where {T<:Number}
    bins = s.bins
    if minimum(bins) == 0
        bins = bins .+ 1 # Julia indices start at one
    end
    return f_0[bins]
end

"""
    decode_estimate(s::LabelSanitizer, f)

Recover the original bins in a deconvolution result `f` after encoding the labels.

**See also:** `encode_labels`, `encode_prior`.
"""
decode_estimate(s::LabelSanitizer, f::Vector{Float64}) where {I<:Integer} =
    decode_estimate(s, convert(Matrix, f'))[:] # treat f like a 1xN matrix

# version for probability matrices (e.g., for DSEA contributions)
function decode_estimate(s::LabelSanitizer, p::Matrix{Float64}) where {I<:Integer}
    bins = s.bins
    if minimum(bins) == 0
        bins = bins .+ 1 # Julia indices start at one
    end
    decoder = Dict(zip(1:length(bins), bins))
    r = zeros(size(p, 1), s.n_bins)
    for (k, v) in decoder
        r[:, v] = p[:, k]
    end
    return r
end

end # module
