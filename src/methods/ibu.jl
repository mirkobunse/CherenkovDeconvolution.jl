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
export IBU

"""
    IBU(binning; kwargs...)

The *Iterative Bayesian Unfolding* deconvolution method, using a `binning` to discretize
the observable features.

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
- `stepsize = DEFAULT_STEPSIZE`
  is the step size taken in every iteration.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, chi2s::Float64, alpha_k::Float64) -> Any` optionally
  called in every iteration.
"""
struct IBU <: DiscreteMethod
    binning :: Binning
    epsilon :: Float64
    f_0 :: Vector{Float64}
    fit_ratios :: Bool
    inspect :: Function
    K :: Int
    n_bins_y :: Int
    smoothing :: Function # TODO smoothing types
    stepsize :: Stepsize
    IBU(binning;
        epsilon    :: Float64  = 0.0,
        f_0        :: Vector{Float64} = Float64[],
        fit_ratios :: Bool     = false,
        inspect    :: Function = (args...) -> nothing,
        K          :: Int64    = 3,
        n_bins_y   :: Int      = -1,
        smoothing  :: Function = Base.identity,
        stepsize   :: Stepsize = DEFAULT_STEPSIZE
    ) = new(binning, epsilon, f_0, fit_ratios, inspect, K, n_bins_y, smoothing, stepsize)
end

binning(ibu::IBU) = ibu.binning
stepsize(ibu::IBU) = ibu.stepsize
expects_normalized_R(ibu::IBU) = !ibu.fit_ratios
expects_normalized_g(ibu::IBU) = true # stick to the default
expected_n_bins_y(ibu::IBU) = ibu.n_bins_y

ibu( x_data  :: AbstractVector{T},
     x_train :: AbstractVector{T},
     y_train :: AbstractVector{T},
     bins_y  :: AbstractVector{T} = 1:maximum(y_train);
     kwargs... ) where T<:Int =
  error("No deprecation redirection implemented")

function deconvolve(
        ibu::IBU,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number}

    # check the arguments and encode the prior
    check_discrete_arguments(R, g)
    f_0 = ibu.f_0
    if length(f_0) > 0 # only need to check if a prior is given
        check_prior(f_0)
        f_0 = DeconvUtil.normalizepdf(f_0)
        if ibu.fit_ratios
            f_0 = f_0 ./ decode_estimate(label_sanitizer, f_trn) # convert to a ratio prior
        end
    elseif ibu.fit_ratios # default prior for ratios
        f_0 = ones(length(f_trn))
    else # set a default uniform prior if none is given
        f_0 = ones(length(f_trn)) ./ length(f_trn)
    end
    f_0 = encode_prior(label_sanitizer, f_0)

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
        f_prev_smooth = k > 1 ? ibu.smoothing(f) : f # do not smooth the initial estimate
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
    return DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f)) # return last estimate
end

# reverse the transfer with Bayes' rule, given the transfer matrix R and the prior f_0
function _ibu_reverse_transfer(R::Matrix{T}, f_0::Vector{Float64}) where T<:Number
    B = zeros(Float64, size(R'))
    for j in 1:size(R, 1)
        B[:, j] = R[j, :] .* f_0 ./ dot(R[j, :], f_0)
    end
    return B
end
