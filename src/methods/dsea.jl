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
export DSEA

"""
    DSEA(classifier; kwargs...)

The *DSEA/DSEA+* deconvolution method, embedding the given `classifier`.

**Keyword arguments**

- `f_0 = ones(m) ./ m`
  defines the prior, which is uniform by default
- `fixweighting = true`
  sets, whether or not the weight update fix is applied. This fix is proposed in my Master's
  thesis and in the corresponding paper.
- `stepsize = DEFAULT_STEPSIZE`
  is the step size taken in every iteration.
- `smoothing = Base.identity`
  is a function that optionally applies smoothing in between iterations.
- `K = 1`
  is the maximum number of iterations.
- `epsilon = 0.0`
  is the minimum symmetric Chi Square distance between iterations. If the actual distance is
  below this threshold, convergence is assumed and the algorithm stops.
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, chi2s::Float64, alpha_k::Float64) -> Any` optionally
  called in every iteration.
- `return_contributions = false`
  sets, whether or not the contributions of individual examples in `X_obs` are returned as
  a tuple together with the deconvolution result.
"""
struct DSEA <: DeconvolutionMethod
    classifier :: Any # hopefully implements the sklearn API
    epsilon :: Float64
    f_0 :: Vector{Float64}
    fixweighting :: Bool
    inspect :: Function
    K :: Int
    n_bins_y :: Int
    return_contributions :: Bool
    smoothing :: Function # TODO smoothing types
    stepsize :: Stepsize
    DSEA(c;
        epsilon      :: Float64  = 0.0,
        f_0          :: Vector{Float64} = Float64[],
        fixweighting :: Bool     = true,
        inspect      :: Function = (args...) -> nothing,
        K            :: Int64    = 1,
        n_bins_y     :: Int      = -1,
        return_contributions :: Bool = false,
        smoothing    :: Function = Base.identity,
        stepsize     :: Stepsize = DEFAULT_STEPSIZE
    ) = new(c, epsilon, f_0, fixweighting, inspect, K, n_bins_y, return_contributions, smoothing, stepsize)
end

# ScikitLearn.jl goes mad when another sub-type of AbstractArray is used.
deconvolve(
        dsea::DSEA,
        X_obs::AbstractArray{T,N},
        X_trn::AbstractArray{T,N},
        y_trn::AbstractVector{I}
        ) where {T,N,I<:Integer} =
    deconvolve(dsea, convert(Array, X_obs), convert(Array, X_trn), convert(Vector, y_trn))

function deconvolve(
        dsea::DSEA,
        X_obs::Array{T,N},
        X_trn::Array{T,N},
        y_trn::Vector{I}
        ) where {T,N,I<:Integer}

    # sanitize and check the arguments
    n_bins_y = max(dsea.n_bins_y, maximum(y_trn)) # number of classes/bins
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
    initialize!(dsea.stepsize, X_obs, X_trn, y_trn)

    # check and encode the prior
    f_0 = dsea.f_0
    if length(f_0) > 0 # only need to check if a prior is given
        check_prior(f_0)
        f_0 = DeconvUtil.normalizepdf(f_0)
    else # set a default uniform prior if none is given
        f_0 = ones(n_bins_y) ./ n_bins_y
    end
    f_0 = encode_prior(label_sanitizer, f_0)

    # initial estimate
    f     = f_0
    f_trn = DeconvUtil.fit_pdf(y_trn, laplace=true) # training pdf with Laplace correction
    w_bin = dsea.fixweighting ? DeconvUtil.normalizepdf(f ./ f_trn, warn=false) : f # bin weights
    w_trn = _dsea_weights(y_trn, w_bin) # instance weights
    dsea.inspect(decode_estimate(label_sanitizer, f), 0, NaN, NaN)
    
    # iterative deconvolution
    proba = Matrix{Float64}(undef, 0, 0) # empty matrix
    alpha_k = Inf
    for k in 1:dsea.K
        f_prev = f
        
        # === update the estimate ===
        train_predict = DeconvUtil.train_and_predict_proba(dsea.classifier)
        proba = train_predict(X_obs, X_trn, y_trn, w_trn)
        f_next = _dsea_reconstruct(proba) # original DSEA reconstruction
        f_step, alpha_k = _dsea_step(
            k,
            decode_estimate(label_sanitizer, f_next), # stepsize assumes original labels
            decode_estimate(label_sanitizer, f_prev),
            alpha_k,
            dsea.stepsize
        )
        f = encode_prior(label_sanitizer, f_step) # encode the result of _dsea_step
        # = = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = DeconvUtil.chi2s(f_prev, f, false) # Chi Square distance between iterations
        @info "DSEA iteration $k/$(dsea.K) uses alpha = $alpha_k (chi2s = $chi2s)"
        dsea.inspect(decode_estimate(label_sanitizer, f), k, chi2s, alpha_k)
        
        # stop when convergence is assumed
        if chi2s < dsea.epsilon # also holds when alpha is zero
            @info "DSEA convergence assumed from chi2s = $chi2s < epsilon = $(dsea.epsilon)"
            break
        end
        
        # == smoothing and reweighting in between iterations ==
        if k < dsea.K
            f = dsea.smoothing(f)
            w_bin = dsea.fixweighting ? DeconvUtil.normalizepdf(f ./ f_trn, warn=false) : f
            w_trn = _dsea_weights(y_trn, w_bin)
        end
        # = = = = = = = = = = = = = = = = = = = = = = = = = = =
        
    end
    
    f_est = decode_estimate(label_sanitizer, f) # revert the encoding of labels
    if !dsea.return_contributions
        return f_est # the default case
    else
        return f_est, decode_estimate(label_sanitizer, proba) # return tuple
    end
    
end

# the weights of training instances are based on the bin weights in w_bin
_dsea_weights(y_trn::Vector{T}, w_bin::Vector{Float64}) where T<:Int =
    max.(w_bin[y_trn], 1/length(y_trn)) # Laplace correction

# the reconstructed estimate is the sum of confidences in each bin
_dsea_reconstruct(proba::Matrix{Float64}) =
    DeconvUtil.normalizepdf(map(i -> sum(proba[:, i]), 1:size(proba, 2)), warn=false)

# the step taken by DSEA+, where alpha may be a constant or a function
function _dsea_step(k::Int64, f::Vector{Float64}, f_prev::Vector{Float64},
                    a_prev::Float64, alpha::Stepsize)
    p_k     = f - f_prev # search direction
    alpha_k = value(alpha, k, p_k, f_prev, a_prev) # function or float
    return  f_prev + alpha_k * p_k,  alpha_k # return a tuple
end

# deprecated syntax
using PyCall
export dsea
function dsea(
        X_obs :: AbstractArray,
        X_trn :: AbstractArray,
        y_trn :: AbstractVector{T},
        train_predict :: Function,
        bins_y :: AbstractVector{T} = 1:maximum(y_trn);
        kwargs...
        ) where T<:Int
    Base.depwarn(join([
        "Deprecated `dsea(data, config)` ignores train_predict and uses GaussianNB; ",
        "please call `deconvolve(DSEA(config), data)` instead"
    ]), :dsea)
    dsea = DSEA(pyimport("sklearn.naive_bayes").GaussianNB(); n_bins_y=length(bins_y), kwargs...) # classifier needed
    return deconvolve(dsea, X_obs, X_trn, y_trn)
end
