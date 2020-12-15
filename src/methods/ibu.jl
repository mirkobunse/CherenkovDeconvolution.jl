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
    ibu(data, train, x, y[, bins_y]; kwargs...)

    ibu(x_data, x_train, y_train[, bins_y]; kwargs...)

    ibu(R, g; kwargs...)


Deconvolve the observed data applying the *Iterative Bayesian Unfolding* trained on the
given training set.

The vectors `x_data`, `x_train`, and `y_train` (or accordingly `data[x]`, `train[x]`, and
`train[y]`) must contain label/observation indices rather than actual values. All expected
indices in `y_train` are optionally provided as `bins_y`. Alternatively, the detector
response matrix `R` and the observed density vector `g` can be given directly.


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
- `alpha = 1.0`
  is the step size taken in every iteration.
  This parameter can be either a constant value or a function with the signature
  `(k::Int, pk::AbstractVector{Float64}, f_prev::AbstractVector{Float64} -> Float`,
  where `f_prev` is the estimate of the previous iteration and `pk` is the direction that
  IBU takes in the current iteration `k`.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, chi2s::Float64, alpha::Float64) -> Any` optionally
  called in every iteration.
- `loggingstream = DevNull`
  is an optional `IO` stream to write log messages to.


**Caution:** According to the value of `fit_ratios`, the keyword argument `f_0` specifies a
ratio prior or a pdf prior, but only in the third form. In the other forms, `f_0` always
specifies a pdf prior.
"""
ibu( data   :: AbstractDataFrame,
     train  :: AbstractDataFrame,
     x      :: Symbol,
     y      :: Symbol,
     bins_y :: AbstractVector = 1:maximum(train[y]);
     kwargs... ) =
  ibu(data[x], train[x], train[y], bins_y; kwargs...) # DataFrame form


# Vector form
ibu( x_data  :: AbstractVector{T},
     x_train :: AbstractVector{T},
     y_train :: AbstractVector{T},
     bins_y  :: AbstractVector{T} = 1:maximum(y_train);
     kwargs... ) where T<:Int =
  _discrete_deconvolution(ibu, x_data, x_train, y_train, bins_y, Dict{Symbol, Any}(kwargs))


function ibu( R :: Matrix{TR},
              g :: Vector{Tg};
              f_0        :: Vector{Float64} = Float64[],
              smoothing  :: Function        = Base.identity,
              K          :: Int             = 3,
              epsilon    :: Float64         = 0.0,
              alpha      :: Union{Float64, Function} = 1.0,
              fit_ratios :: Bool            = false,
              inspect    :: Function        = (args...) -> nothing,
              loggingstream :: IO = devnull,
              kwargs... ) where {TR<:Number, Tg<:Number}
    
    # check arguments
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    end
    
    if loggingstream != devnull
        @warn "The argument 'loggingstream' is deprecated in v0.1.0. Use the 'with_logger' functionality of julia-0.7 and above." _group=:depwarn
    end
    
    # initial estimate
    f = _check_prior(f_0, size(R, 2), fit_ratios) # do not normalize if ratios are fitted
    inspect(f, 0, NaN, NaN) # inspect prior
    
    # iterative Bayesian deconvolution
    for k in 1:K
        
        # == smoothing in between iterations ==
        f_prev_smooth = k > 1 ? smoothing(f) : f # do not smooth the initial estimate
        f_prev = f # unsmoothed estimate required for convergence check
        # = = = = = = = = = = = = = = = = = = =
        
        # === apply Bayes' rule ===
        f = _ibu_reverse_transfer(R, f_prev_smooth) * g
        if !fit_ratios
            f = DeconvUtil.normalizepdf(f, warn=false)
        end
        # = = = = = = = = = = = = =
        
        # == apply stepsize update ==
        pk = f - f_prev_smooth
        alphak = typeof(alpha) == Float64 ? alpha : alpha(k, pk, f_prev_smooth)
        f = f_prev_smooth + alphak * pk
        # = = = = = = = = = = = = =

        # monitor progress
        chi2s = DeconvUtil.chi2s(f_prev, f, false) # Chi Square distance between iterations
        @debug "IBU iteration $k/$K (chi2s = $chi2s) with alpha = $(alphak)"
        inspect(f, k, chi2s, alphak)
        
        # stop when convergence is assumed
        if chi2s < epsilon
            @debug "IBU convergence assumed from chi2s = $chi2s < epsilon = $epsilon"
            break
        end
        
    end
    
    return f # return last estimate
    
end

# reverse the transfer with Bayes' rule, given the transfer matrix R and the prior f_0
function _ibu_reverse_transfer(R::Matrix{T}, f_0::Vector{Float64}) where T<:Number
    B = zeros(Float64, size(R'))
    for j in 1:size(R, 1)
        B[:, j] = R[j, :] .* f_0 ./ dot(R[j, :], f_0)
    end
    return B
end

