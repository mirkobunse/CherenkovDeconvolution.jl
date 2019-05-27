# 
# CherenkovDeconvolution.jl
# Copyright 2018, 2019 Mirko Bunse
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
    ibu(data, train, x, y[, bins_y; kwargs...])

Iterative Bayesian Unfolding of the target distribution in the DataFrame `data`. The
deconvolution is inferred from the DataFrame `train`, where the target column `y` and the
observable column `x` are given.

This function wraps `ibu(R, g; kwargs...)`, constructing `R` and `g` from the examples in
the two DataFrames.
"""
ibu(data::AbstractDataFrame, train::AbstractDataFrame, x::Symbol, y::Symbol,
    bins_y::AbstractArray = 1:maximum(train[y]); kwargs...) =
  ibu(data[x], train[x], train[y], bins_y; kwargs...)

"""
    ibu(x_data, x_train, y_train[, bins_y; kwargs...])

Iterative Bayesian Unfolding of the target distribution, given the observations in the
one-dimensional array `x_data`.

The deconvolution is inferred from `x_train` and `y_train`. Both of these arrays have to be
discrete, i.e., they must contain indices instead of actual values. All expected label
indices (for cases where `y_train` may not contain some of the indices) are optionally
provided as `bins_y`.

This function wraps `ibu(R, g; kwargs...)`, constructing `R` and `g` from the examples in
the three arrays.
"""
function ibu(x_data::AbstractArray{T, 1},
             x_train::AbstractArray{T, 1},
             y_train::AbstractArray{T, 1},
             bins::AbstractArray{T, 1} = 1:maximum(y_train);
             kwargs...) where T<:Int
                     
    bins_x = 1:maximum(vcat(x_data, x_train)) # no need to provide this as an argument
    
    # recode indices
    recode_dict, y_train = _recode_indices(bins, y_train)
    _, x_train, x_data   = _recode_indices(bins_x, x_train, x_data)
    kwargs_dict = Dict{Symbol, Any}(kwargs)
    if haskey(kwargs_dict, :f_0)
        kwargs_dict[:f_0] = _check_prior(kwargs_dict[:f_0], recode_dict)
    end
    if haskey(kwargs_dict, :inspect)
        fun = kwargs_dict[:inspect] # inspection function
        kwargs_dict[:inspect] = (f, args...) -> fun(_recode_result(f, recode_dict), args...)
    end
    
    # prepare arguments
    R = Util.fit_R(y_train, x_train, bins_y = bins, bins_x = bins_x)
    g = Util.fit_pdf(x_data, bins_x)
    
    # deconvolve
    f = ibu(R, g; kwargs_dict...)
    return _recode_result(f, recode_dict) # revert recoding of labels
    
end

"""
    ibu(R, g; kwargs...)

Iterative Bayesian Unfolding with the detector response matrix `R` and the observable
density function `g`.

**Keyword arguments**

- `f_0 = ones(m) ./ m`
  defines the prior, which is uniform by default.
- `smoothing = Base.identity`
  is a function that optionally applies smoothing in between iterations. The operation is
  neither applied to the initial prior, nor to the final result. The function `inspect` is
  called before the smoothing is performed.
- `K = 3`
  is the maximum number of iterations.
- `epsilon = 0.0`
  is the minimum symmetric Chi Square distance between iterations. If the actual distance is
  below this threshold, convergence is assumed and the algorithm stops.
- `inspect = nothing`
  is a function `(f_k::Array, k::Int, chi2s::Float64) -> Any` optionally called in every
  iteration.
"""
function ibu(R::Matrix{Float64}, g::Array{T, 1};
             f_0::Array{Float64, 1} = Float64[],
             smoothing::Function = Base.identity,
             K::Int = 3,
             epsilon::Float64 = 0.0,
             inspect::Function = (args...) -> nothing,
             loggingstream::IO = devnull) where T<:Number
    
    # check arguments
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    end
    f_0 = _check_prior(f_0, size(R, 2))
    
    if loggingstream != devnull
        @warn "The argument 'loggingstream' is deprecated in v0.1.0. Use the 'with_logger' functionality of julia-0.7 and above." _group=:depwarn
    end
    
    # initial estimate
    f = Util.normalizepdf(f_0)
    inspect(f, 0, NaN) # inspect prior
    
    # iterative Bayesian deconvolution
    for k in 1:K
        
        # == smoothing in between iterations ==
        f_prev_smooth = k > 1 ? smoothing(f) : f # do not smooth the initial estimate
        f_prev = f # unsmoothed estimate required for convergence check
        # = = = = = = = = = = = = = = = = = = =
        
        # === apply Bayes' rule ===
        f = Util.normalizepdf(_ibu_reverse_transfer(R, f_prev_smooth) * g)
        # = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = Util.chi2s(f_prev, f, false) # Chi Square distance between iterations
        @debug "IBU iteration $k/$K (chi2s = $chi2s)"
        inspect(f, k, chi2s)
        
        # stop when convergence is assumed
        if chi2s < epsilon
            @debug "IBU convergence assumed from chi2s = $chi2s < epsilon = $epsilon"
            break
        end
        
    end
    
    return f # return last estimate
    
end

# reverse the transfer with Bayes' rule, given the transfer matrix R and the prior f_0
function _ibu_reverse_transfer(R::Matrix{Float64}, f_0::Array{Float64, 1})
    B = zero(R')
    for j in 1:size(R, 1)
        B[:, j] = R[j, :] .* f_0 ./ dot(R[j, :], f_0)
    end
    return B
end

