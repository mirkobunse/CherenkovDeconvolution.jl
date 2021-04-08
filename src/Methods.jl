# 
# CherenkovDeconvolution.jl
# Copyright 2018, 2019, 2020 Mirko Bunse
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

using DataFrames, LinearAlgebra, Optim
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

"""
    LabelSanitizer(y_trn, n_bins=maximum(y_trn))

A sanitizer that

- encodes labels and priors so that none of the resulting bins is empty.
- decodes deconvolution results to recover the original (possibly empty) bins.

**See also:** `encode_labels`, `encode_prior`, `decode_estimate`.
"""
struct LabelSanitizer
    bins::Vector{Int} # bins that actually appear
    n_bins::Int # assumed number of bins
    LabelSanitizer(y_trn::AbstractVector{I}, n_bins::Int=maximum(y_trn)) where {I<:Integer} =
        new(sort(unique(y_trn)), n_bins)
end

"""
    abstract type DeconvolutionMethod

The supertype of all deconvolution methods.
"""
abstract type DeconvolutionMethod end

"""
    deconvolve(m, X_obs, X_trn, y_trn)

Deconvolve the observed features in `X_obs` with the deconvolution method `m`
trained on the features `X_trn` and the corresponding labels `y_trn`.
"""
deconvolve(
        m::DeconvolutionMethod,
        X_obs::AbstractArray{T,N},
        X_trn::AbstractArray{T,N},
        y_trn::AbstractVector{I}
        ) where {T,N,I<:Integer} =
    throw(ArgumentError("Implementation missing for $(typeof(m))")) # must be implemented for sub-types

"""
    abstract type DiscreteMethod <: DeconvolutionMethod

The supertype of all classical deconvolution methods which estimate the density
function `f` from a transfer matrix `R` and an observed density `g`.
"""
abstract type DiscreteMethod <: DeconvolutionMethod end

# discrete methods actually deconvolve from R and g, so the general API must wrap them
function deconvolve(
        m::DiscreteMethod,
        X_obs::AbstractArray,
        X_trn::AbstractArray,
        y_trn::AbstractVector{I}
        ) where {I<:Integer}

    # sanitize and check the arguments
    n_bins_y = max(expected_n_bins_y(m), maximum(y_trn)) # number of classes/bins
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

    # discretize the problem statement into a system of linear equations
    d = BinningDiscretizer(binning(m), X_trn, y_trn) # fit the binning strategy with labeled data
    x_obs = encode(d, X_obs) # apply it to the feature vectors
    x_trn = encode(d, X_trn)
    R = DeconvUtil.fit_R(y_trn, x_trn; bins_x=bins(d), normalize=expects_normalized_R(m))
    g = DeconvUtil.fit_pdf(x_obs, bins(d); normalize=expects_normalized_g(m))
    f_trn = DeconvUtil.fit_pdf(y_trn)

    # call the actual solver (IBU, RUN, etc)
    return deconvolve(m, R, g, label_sanitizer, f_trn)
end

# required API for discrete methods
deconvolve(
        m::DiscreteMethod,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number} =
    throw(ArgumentError("Implementation missing for $(typeof(m))")) # must be implemented for sub-types
binning(m::DiscreteMethod) = throw(ArgumentError("Implementation missing for $(typeof(m))"))
expects_normalized_R(m::DiscreteMethod) = false # default
expects_normalized_g(m::DiscreteMethod) = true # default
expected_n_bins_y(m::DiscreteMethod) = 0

export dsea, ibu, p_run, run, svd

# deconvolution methods
include("methods/svd.jl")
include("methods/run.jl")
include("methods/prun.jl")
include("methods/ibu.jl")
include("methods/dsea.jl")

# wrapper for classical algorithms (e.g. run or ibu) to set up R and g, then calling the solver
function _discrete_deconvolution( solver  :: Function,
                                  x_data  :: AbstractVector{T},
                                  x_train :: AbstractVector{T},
                                  y_train :: AbstractVector{T},
                                  bins_y  :: AbstractVector{T},
                                  kw_dict :: Dict{Symbol, Any};
                                  normalize_g::Bool=true ) where T<:Int
    # recode indices
    recode_dict, y_train = _recode_indices(bins_y, y_train)
    _, x_data, x_train   = _recode_indices(1:maximum(vcat(x_data, x_train)), x_data, x_train)
    
    # prepare the arguments for the solver
    bins_x = 1:maximum(vcat(x_data, x_train)) # ensure same bins in R and g
    fit_ratios = get(kw_dict, :fit_ratios, false) # ratios fitted instead of pdfs?
    R = DeconvUtil.fit_R(y_train, x_train, bins_x = bins_x, normalize = !fit_ratios)
    g = DeconvUtil.fit_pdf(x_data, bins_x, normalize = normalize_g) # absolute counts instead of pdf
    
    # recode the prior (if specified)
    if haskey(kw_dict, :f_0)
        f_0 = _check_prior(kw_dict[:f_0], recode_dict) # also normalizes f_0
        if fit_ratios
            f_0 = f_0 ./ DeconvUtil.fit_pdf(y_train) # pdf prior -> ratio prior
        end
        kw_dict[:f_0] = f_0 # update
    elseif fit_ratios
        kw_dict[:f_0] = ones(size(R, 2)) ./ DeconvUtil.fit_pdf(y_train) # uniform prior instead of f_train
    end
    
    # inspect with original coding of labels
    if haskey(kw_dict, :inspect)
        inspect = kw_dict[:inspect] # inspection function
        kw_dict[:inspect] = (f_est, args...) -> begin
            if fit_ratios
                f_est = f_est .* DeconvUtil.fit_pdf(y_train) # ratio solution -> pdf solution
            end
            inspect(DeconvUtil.normalizepdf(_recode_result(f_est, recode_dict), warn=false), args...)
        end
    end
    
    # call the solver (ibu, run,...)
    f_est = solver(R, g; kw_dict...)
    if fit_ratios
        f_est = f_est .* DeconvUtil.fit_pdf(y_train) # ratio solution -> pdf solution
    end
    return DeconvUtil.normalizepdf(_recode_result(f_est, recode_dict)) # revert recoding of labels
end



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
check_arguments(X_obs::AbstractArray{T,N}, X_trn::AbstractArray{T,N}, y_trn::AbstractVector{I}) where {T,N,I<:Integer} =
    if size(X_obs, 2) != size(X_trn, 2)
        throw(ArgumentError("X_obs and X_trn do not have the same number of features"))
    elseif size(X_trn, 1) != length(y_trn)
        throw(ArgumentError("X_trn and y_trn do not represent the same number of samples"))
    elseif size(X_trn, 1) == 0
        throw(ArgumentError("There are no samples in the training set (X_trn, y_trn)"))
    elseif size(X_trn, 2) == 0
        throw(ArgumentError("There are no features in the data (X_obs, X_trn)"))
    elseif all(y_trn .== y_trn[1])
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
encode_prior(s::LabelSanitizer, f_0::AbstractVector{T}) where {T<:Number} = f_0[s.bins]

"""
    decode_estimate(s::LabelSanitizer, f)

Recover the original bins in a deconvolution result `f` after encoding the labels.

**See also:** `encode_labels`, `encode_prior`.
"""
decode_estimate(s::LabelSanitizer, f::Vector{Float64}) where {I<:Integer} =
    decode_estimate(s, convert(Matrix, f'))[:] # treat f like a 1xN matrix

# version for probability matrices (e.g., for DSEA contributions)
function decode_estimate(s::LabelSanitizer, p::Matrix{Float64}) where {I<:Integer}
    decoder = Dict(zip(1:length(s.bins), s.bins))
    r = zeros(size(p, 1), s.n_bins)
    for (k, v) in decoder
        r[:, v] = p[:, k]
    end
    return r
end

# check and repair the f_0 argument
function _check_prior(f_0::Vector{Float64}, m::Int64, fit_ratios::Bool=false)
    Base.depwarn("`_check_prior` is deprecated, use `check_prior` with custom ratio handling instead.", :_check_prior)
    if length(f_0) == 0
        return fit_ratios ? ones(m) : ones(m) ./ m
    elseif length(f_0) != m
        throw(DimensionMismatch("dim(f_0) = $(length(f_0)) != $m, the number of classes"))
    elseif fit_ratios # f_0 is provided and alright
        return f_0
    else
        return DeconvUtil.normalizepdf(f_0) # ensure pdf (default case)
    end
end

_check_prior(f_0::Vector{Float64}, recode_dict::Dict{T, T}) where T<:Int =
    _check_prior(length(f_0) > 0 ? f_0[sort(setdiff(collect(values(recode_dict)), [-1]))] : f_0, length(recode_dict)-1 )

function _recode_indices(bins::AbstractVector{T}, inds::AbstractVector{T}...) where T<:Int
    Base.depwarn("`_recode_indices` is deprecated, use `encode_labels` instead.", :_recode_indices)
    s = LabelSanitizer(vcat(inds...), maximum(bins))
    d = Dict(zip(1:length(s.bins), s.bins)) # decoder
    d[-1] = s.n_bins # the highest assumed bin
    return d, [encode_labels(s, i) for i in inds]...
end

# recode a deconvolution result by reverting the initial recoding of the data
_recode_result(f::Vector{Float64}, recode_dict::Dict{T, T}) where T<:Int =
    _recode_result(convert(Matrix, f'), recode_dict)[:] # treat f like a 1xN matrix

# like above but for probability matrices (used if DSEA returns contributions)
function _recode_result(proba::Matrix{Float64}, recode_dict::Dict{T, T}) where T<:Int
    Base.depwarn("`_recode_result` is deprecated, use `decode_estimate` instead.", :_recode_result)
    r = zeros(Float64, size(proba, 1), maximum(values(recode_dict)))
    for (k, v) in recode_dict
        if k != -1
            r[:, v] = proba[:, k]
        end # else, the key was just included to store the maximum value
    end
    return r
end

end # module
