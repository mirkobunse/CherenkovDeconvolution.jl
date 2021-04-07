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
using ..DeconvUtil, ..Binnings
import ..DEFAULT_STEPSIZE, ..Stepsize, ..stepsize

export DeconvolutionMethod, DiscreteMethod

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
    deconvolve(m, X_obs, X_trn, y_trn)

Deconvolve the observed features in `X_obs` with the deconvolution method `m`
trained on the features `X_trn` and the corresponding labels `y_trn`.
"""
deconvolve(m::DeconvolutionMethod, X_obs::AbstractArray, X_trn::AbstractArray, y_trn::AbstractVector{I}) where I<:Integer =
    throw(ArgumentError("Implementation missing for $(typeof(m))")) # must be implemented for sub-types

# discrete methods actually deconvolve from R and g, so the general API must wrap them
function deconvolve(m::DiscreteMethod, X_obs::AbstractArray, X_trn::AbstractArray, y_trn::AbstractVector{I}) where I<:Integer
    d = BinningDiscretizer(binning(m), X_trn, y_trn) # fit the binning strategy with labeled data
    x_obs = encode(d, X_obs) # apply it to the feature vectors
    x_trn = encode(d, X_trn)
    R = DeconvUtil.fit_R(y_trn, x_trn; bins_x=bins(d), normalize=fits_ratios(m))
    g = DeconvUtil.fit_pdf(x_obs, bins(d); normalize=normalizes_g(m))
    return deconvolve(m, R, g)
end

# required API for discrete methods
deconvolve(m::DiscreteMethod, R::Matrix{T_R}, g::Vector{T_g}) where {T_R<:Number, T_g<:Number} =
    throw(ArgumentError("Implementation missing for $(typeof(m))")) # must be implemented for sub-types
binning(m::DiscreteMethod) = throw(ArgumentError("Implementation missing for $(typeof(m))"))
fits_ratios(m::DiscreteMethod) = throw(ArgumentError("Implementation missing for $(typeof(m))"))
normalizes_g(m::DiscreteMethod) = throw(ArgumentError("Implementation missing for $(typeof(m))"))
prior(m::DiscreteMethod) = nothing # a prior is optional

export dsea, ibu, p_run, run, svd

# deconvolution methods
include("methods/svd.jl")
include("methods/run.jl")
include("methods/p_run.jl")
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

# check and repair the f_0 argument
function _check_prior(f_0::Vector{Float64}, m::Int64, fit_ratios::Bool=false)
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

# recode indices to resemble a unit range (no missing labels in between)
function _recode_indices(bins::AbstractVector{T}, inds::AbstractVector{T}...) where T<:Int
    
    # recode the training set
    inds_bins = sort(unique(vcat(inds...)))
    inds_dict = Dict(zip(inds_bins, 1:length(inds_bins)))
    inds_rec  = map(ind -> map(i -> inds_dict[i], ind), inds)
    
    # set up reverse recoding applied in _recode_result
    recode_dict = Dict(zip(values(inds_dict), keys(inds_dict))) # map from values to keys
    recode_dict[-1] = maximum(bins) # the highest actual bin (may not be in y_train)
    
    return recode_dict, inds_rec...
    
end

# recode a deconvolution result by reverting the initial recoding of the data
_recode_result(f::Vector{Float64}, recode_dict::Dict{T, T}) where T<:Int =
    _recode_result(convert(Matrix, f'), recode_dict)[:] # treat f like a 1xN matrix

# like above but for probability matrices (used if DSEA returns contributions)
function _recode_result(proba::Matrix{Float64}, recode_dict::Dict{T, T}) where T<:Int
    r = zeros(Float64, size(proba, 1), maximum(values(recode_dict)))
    for (k, v) in recode_dict
        if k != -1
            r[:, v] = proba[:, k]
        end # else, the key was just included to store the maximum value
    end
    return r
end

end # module
