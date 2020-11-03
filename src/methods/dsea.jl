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
    dsea(data, train, y, train_predict[, bins_y, features]; kwargs...)

    dsea(X_data, X_train, y_train, train_predict[, bins_y]; kwargs...)


Deconvolve the observed data with *DSEA/DSEA+* trained on the given training set.

The data is provided as feature matrices `X_data`, `X_train` and the label vector `y_train`
(or accordingly `data[features]`, `train[features]`, and `train[y]`). Here, `y_train` must
contain label indices rather than actual values. All expected indices are optionally provided
as `bins_y`.

The function object `train_predict(X_data, X_train, y_train, w_train) -> Matrix` trains and
applies a classifier to obtain a confidence matrix.


**Keyword arguments**

- `f_0 = ones(m) ./ m`
  defines the prior, which is uniform by default
- `fixweighting = true`
  sets, whether or not the weight update fix is applied. This fix is proposed in my Master's
  thesis and in the corresponding paper.
- `alpha = 1.0`
  is the step size taken in every iteration.
  This parameter can be either a constant value or a function with the signature
  `(k::Int, pk::AbstractVector{Float64}, f_prev::AbstractVector{Float64} -> Float`,
  where `f_prev` is the estimate of the previous iteration and `pk` is the direction that
  DSEA takes in the current iteration `k`.
- `smoothing = Base.identity`
  is a function that optionally applies smoothing in between iterations
- `K = 1`
  is the maximum number of iterations.
- `epsilon = 0.0`
  is the minimum symmetric Chi Square distance between iterations. If the actual distance is
  below this threshold, convergence is assumed and the algorithm stops.
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, chi2s::Float64, alpha::Float64) -> Any` optionally
  called in every iteration.
- `return_contributions = false`
  sets, whether or not the contributions of individual examples in `X_data` are returned as
  a tuple together with the deconvolution result.
- `features = setdiff(names(train), [y])`
  specifies which columns in `data` and `train` to be used as features - only applicable to
  the first form of this function.
"""
dsea( data          :: AbstractDataFrame,
      train         :: AbstractDataFrame,
      y             :: Symbol,
      train_predict :: Function,
      bins_y        :: AbstractVector{T} = 1:maximum(train[y]);
      features      :: AbstractVector{Symbol} = setdiff(names(train), [y]),
      kwargs... ) where T<:Int =
  dsea(DeconvUtil.df2X(data, features),
       DeconvUtil.df2Xy(train, y, features)...,
       train_predict,
       bins_y;
       kwargs...) # DataFrame form


# Vector/Matrix form
# 
# Here, X_data, X_train, and y_train are only converted to actual Array objects because
# ScikitLearn.jl goes mad when some of the other sub-types of AbstractArray are used. The
# actual implementation is below.
dsea( X_data        :: AbstractArray,
      X_train       :: AbstractArray,
      y_train       :: AbstractVector{T},
      train_predict :: Function,
      bins_y        :: AbstractVector{T} = 1:maximum(y_train);
      kwargs... ) where T<:Int =
  _dsea(convert(Array, X_data), convert(Array, X_train), convert(Vector, y_train),
        train_predict, convert(Vector, bins_y); kwargs...)


function _dsea(X_data        :: Array,
               X_train       :: Array,
               y_train       :: Vector{T},
               train_predict :: Function,
               bins_y        :: Vector{T} = 1:maximum(y_train);
               f_0           :: Vector{Float64} = Float64[],
               alpha         :: Union{Float64, Function} = 1.0,
               fixweighting  :: Bool     = true,
               smoothing     :: Function = Base.identity,
               K             :: Int64    = 1,
               epsilon       :: Float64  = 0.0,
               inspect       :: Function = (args...) -> nothing,
               loggingstream :: IO       = devnull,
               return_contributions :: Bool = false) where T<:Int
    
    # recode labels and check arguments
    if all(y_train .== y_train[1])
        f_est = zeros(length(bins_y))
        f_est[bins_y .== y_train[1]] .= 1
        @warn "Only one label in the training set, returning a trivial estimate" f_est
        return f_est
    end
    recode_dict, y_train = _recode_indices(bins_y, y_train)
    if size(X_data, 2) != size(X_train, 2)
        throw(ArgumentError("X_data and X_train do not have the same number of features"))
    end
    f_0 = _check_prior(f_0, recode_dict)
    
    if loggingstream != devnull
        @warn "The argument 'loggingstream' is deprecated in v0.1.0. Use the 'with_logger' functionality of julia-0.7 and above." _group=:depwarn
    end
    
    # initial estimate
    f       = f_0
    f_train = DeconvUtil.fit_pdf(y_train, laplace=true)                            # training pdf with Laplace correction
    w_bin   = fixweighting ? DeconvUtil.normalizepdf(f ./ f_train, warn=false) : f # bin weights
    w_train = _dsea_weights(y_train, w_bin)                                  # instance weights
    inspect(_recode_result(f, recode_dict), 0, NaN, NaN)
    
    # iterative deconvolution
    proba = Matrix{Float64}(undef, 0, 0) # empty matrix
    for k in 1:K
        f_prev = f
        
        # === update the estimate ===
        proba     = train_predict(X_data, X_train, y_train, w_train)
        f_next    = _dsea_reconstruct(proba) # original DSEA reconstruction
        f, alphak = _dsea_step( k,
                                _recode_result(f_next, recode_dict),
                                _recode_result(f_prev, recode_dict),
                                alpha ) # step size function assumes original coding
        f = _check_prior(f, recode_dict) # re-code result of _dsea_step
        # = = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = DeconvUtil.chi2s(f_prev, f, false) # Chi Square distance between iterations
        @info "DSEA iteration $k/$K uses alpha = $alphak (chi2s = $chi2s)"
        inspect(_recode_result(f, recode_dict), k, chi2s, alphak)
        
        # stop when convergence is assumed
        if chi2s < epsilon # also holds when alpha is zero
            @info "DSEA convergence assumed from chi2s = $chi2s < epsilon = $epsilon"
            break
        end
        
        # == smoothing and reweighting in between iterations ==
        if k < K
            f = smoothing(f)
            w_bin   = fixweighting ? DeconvUtil.normalizepdf(f ./ f_train, warn=false) : f
            w_train = _dsea_weights(y_train, w_bin)
        end
        # = = = = = = = = = = = = = = = = = = = = = = = = = = =
        
    end
    
    f_rec = _recode_result(f, recode_dict) # revert recoding of labels
    if !return_contributions # the default case
        return f_rec
    else
        return f_rec, _recode_result(proba, recode_dict) # return tuple
    end
    
end

# the weights of training instances are based on the bin weights in w_bin
_dsea_weights(y_train::Vector{T}, w_bin::Vector{Float64}) where T<:Int =
    max.(w_bin[y_train], 1/length(y_train)) # Laplace correction

# the reconstructed estimate is the sum of confidences in each bin
_dsea_reconstruct(proba::Matrix{Float64}) =
    DeconvUtil.normalizepdf(map(i -> sum(proba[:, i]), 1:size(proba, 2)), warn=false)

# the step taken by DSEA+, where alpha may be a constant or a function
function _dsea_step(k::Int64, f::Vector{Float64}, f_prev::Vector{Float64},
                    alpha::Union{Float64, Function})
    pk     = f - f_prev                                              # search direction
    alphak = typeof(alpha) == Float64 ? alpha : alpha(k, pk, f_prev) # function or float
    return  f_prev + alphak * pk,  alphak                            # return tuple
end
