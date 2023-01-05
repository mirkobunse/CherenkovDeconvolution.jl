# 
# CherenkovDeconvolution.jl
# Copyright 2018-2023 Mirko Bunse
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
export IBU

"""
    IBU(binning; kwargs...)

The *Iterative Bayesian Unfolding* deconvolution method, using a `binning` to discretize
the observable features.

**Keyword arguments**

- `f_0 = ones(m) ./ m`
  defines the prior, which is uniform by default.
- `smoothing = NoSmoothing()`
  is an object that optionally applies smoothing in between iterations. The operation is
  neither applied to the initial prior, nor to the final result. The function `inspect` is
  called before the smoothing is performed.
- `K = 3`
  is the maximum number of iterations.
- `epsilon = 0.0`
  is the minimum symmetric Chi Square distance between iterations. If the actual distance is
  below this threshold, convergence is assumed and the algorithm stops.
- `stepsize = DEFAULT_STEPSIZE`
  is the step size taken in every iteration.
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, chi2s::Float64, alpha_k::Float64) -> Any` optionally
  called in every iteration.
- `warn = true`
  determines whether warnings about negative values are emitted during normalization.
- `fit_ratios = false` (**discouraged**)
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.
"""
struct IBU <: DiscreteMethod
    binning :: Binning
    epsilon :: Float64
    f_0 :: Union{Vector{Float64},Nothing}
    fit_ratios :: Bool
    inspect :: Function
    K :: Int
    n_bins_y :: Int
    smoothing :: Smoothing
    stepsize :: Stepsize
    warn :: Bool
    function IBU(binning :: Binning;
            epsilon    :: Float64  = 0.0,
            f_0        :: Union{Vector{Float64},Nothing} = nothing,
            fit_ratios :: Bool     = false,
            inspect    :: Function = (args...) -> nothing,
            K          :: Int64    = 3,
            n_bins_y   :: Int      = -1,
            smoothing  :: Smoothing = NoSmoothing(),
            stepsize   :: Stepsize = DEFAULT_STEPSIZE,
            warn       :: Bool     = true)
        if fit_ratios
            @warn "fit_ratios = true is an experimental feature that is discouraged for IBU"
        end
        return new(binning, epsilon, f_0, fit_ratios, inspect, K, n_bins_y, smoothing, stepsize, warn)
    end
end

binning(ibu::IBU) = ibu.binning
stepsize(ibu::IBU) = ibu.stepsize
prior(ibu::IBU) = ibu.f_0
expects_normalized_R(ibu::IBU) = !ibu.fit_ratios
expects_normalized_g(ibu::IBU) = true # stick to the default
expected_n_bins_y(ibu::IBU) = ibu.n_bins_y

function deconvolve(
        ibu::IBU,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f},
        f_0::Union{Vector{T_f},Nothing}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number}

    # check the arguments and set the optional prior
    check_discrete_arguments(R, g)
    if f_0 == nothing
        if ibu.fit_ratios # default prior for ratios
            f_0 = ones(length(f_trn))
        else # default uniform prior
            f_0 = ones(length(f_trn)) ./ length(f_trn)
        end
    elseif ibu.fit_ratios
        f_0 = f_0 ./ f_trn # convert to a ratio prior
    end

    # the initial estimate
    f = f_0
    f_inspect = f
    if ibu.fit_ratios
        f_inspect = f_inspect .* f_trn # convert a ratio solution to a pdf solution
    end
    ibu.inspect(DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f_inspect), warn=false), 0, NaN, NaN)

    # iterative Bayesian deconvolution
    alpha_k = Inf
    for k in 1:ibu.K

        # == smoothing in between iterations ==
        f_prev_smooth = k > 1 ? DeconvUtil.normalizepdf(apply(ibu.smoothing, f)) : f # do not smooth the initial estimate
        f_prev = f # unsmoothed estimate required for convergence check
        # = = = = = = = = = = = = = = = = = = =

        # === apply Bayes' rule ===
        f = _ibu_reverse_transfer(R, f_prev_smooth) * g
        if !ibu.fit_ratios
            f = DeconvUtil.normalizepdf(f, warn=false)
        end
        # = = = = = = = = = = = = =

        # == apply stepsize update ==
        p_k = f - f_prev_smooth
        alpha_k = value(ibu.stepsize, k, p_k, f_prev_smooth, alpha_k)
        f = f_prev_smooth + alpha_k * p_k
        # = = = = = = = = = = = = =

        # monitor progress
        chi2s = DeconvUtil.chi2s(f_prev, f, false) # Chi Square distance between iterations
        @debug "IBU iteration $k/$(ibu.K) (chi2s = $chi2s) with alpha = $(alpha_k)"
        f_inspect = f
        if ibu.fit_ratios
            f_inspect = f_inspect .* f_trn # convert a ratio solution to a pdf solution
        end
        ibu.inspect(DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f_inspect), warn=false), k, chi2s, alpha_k)

        # stop when convergence is assumed
        if chi2s < ibu.epsilon
            @debug "IBU convergence assumed from chi2s = $chi2s < epsilon = $(ibu.epsilon)"
            break
        end
    end

    if ibu.fit_ratios
        f = f .* f_trn # convert a ratio solution to a pdf solution
    end
    return DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f), warn=ibu.warn) # return last estimate
end

# reverse the transfer with Bayes' rule, given the transfer matrix R and the prior f_0
function _ibu_reverse_transfer(R::Matrix{T}, f_0::Vector{Float64}) where T<:Number
    B = zeros(Float64, size(R'))
    for j in 1:size(R, 1)
        B[:, j] = R[j, :] .* f_0 ./ dot(R[j, :], f_0)
    end
    return B
end

# deprecated syntax (the IdentityBinning is defined in src/methods/run.jl)
export ibu
function ibu(
        x_obs  :: AbstractVector{T},
        x_trn  :: AbstractVector{T},
        y_trn  :: AbstractVector{T},
        bins_y :: AbstractVector{T} = 1:maximum(y_trn);
        kwargs...
        ) where T<:Int
    Base.depwarn(join([
        "`ibu(data, config)` is deprecated; ",
        "please call `deconvolve(IBU(config), data)` instead"
    ]), :ibu)
    ibu = IBU(IdentityBinning(); n_bins_y=length(bins_y), kwargs...)
    return deconvolve(
        ibu,
        reshape(x_obs, (length(x_obs), 1)), # treat as a matrix
        reshape(x_trn, (length(x_trn), 1)),
        y_trn
    )
end
