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
    p_run(data, train, x, y[, m_y]; kwargs...)

    p_run(x_data, x_train, y_train[, m_y]; kwargs...)

    p_run(R, g; kwargs...)


Deconvolve the observed data applying the *Regularized Unfolding* trained on the given
training set.

The vectors `x_data`, `x_train`, and `y_train` (or accordingly `data[x]`, `train[x]`, and
`train[y]`) must contain label/observation indices rather than actual values. All expected
indices in `y_train` are optionally provided as `m_y`. Alternatively, the detector
response matrix `R` and the observed density vector `g` can be given directly.


**Keyword arguments**

- `K = 100`
  is the maximum number of iterations.
- `epsilon = 1e-6`
  is the minimum difference in the loss function between iterations. p_RUN stops when the
  absolute loss difference drops below `epsilon`.
- `tau = 0.0` 
   determines the regularisation strength.
- `x0 = fill(1.0, size(R, 2))`
   Starting point for the interior-point Newton optimization
- `acceptance_correction = nothing` 
  is a tuple of functions (ac(d), inv_ac(d)) representing the acceptance correction
  ac and its inverse operation inv_ac for a data set d.
- `ac_regularisation = true` 
   decides whether acceptance correction is taken into account for regularisation.
   Requires `acceptance_correction` != nothing.
- `evaluate_ac_spectrum = true`
  decides whether spectrums will be acceptance corrected. Requires `acceptance_correction` != nothing. 
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, ldiff::Float64) -> Any` optionally
  called in every iteration.
- `loggingstream = devnull`
  is an optional `IO` stream to write log messages to.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.


**Caution:** According to the value of `fit_ratios`, the keyword argument `f_0` specifies a
ratio prior or a pdf prior, but only in the third form. In the other forms, `f_0` always
specifies a pdf prior.
"""
p_run( data   :: AbstractDataFrame,
     train  :: AbstractDataFrame,
     x      :: Symbol,
     y      :: Symbol,
     m_y :: AbstractVector = 1:maximum(train[y]);
     kwargs... ) =
  p_run(data[x], train[x], train[y], m_y; kwargs...) # DataFrame form


# Vector form
p_run( x_data  :: AbstractVector{T},
       x_train :: AbstractVector{T},
       y_train :: AbstractVector{T},
       m_y  :: AbstractVector{T} = 1:maximum(y_train);
       kwargs... ) where T<:Int =
    _discrete_deconvolution(p_run, x_data, x_train, y_train, m_y, Dict{Symbol, Any}(kwargs), normalize_g=false)


function p_run( R :: Matrix{TR},
	          g :: Vector{Tg};
            K       :: Int      = 100,
            epsilon :: Float64  = 1e-6,
            tau     :: Float64  = 0.0, 
            x0      :: Vector{Float64}  = fill(1.0, size(R, 2)), 
            acceptance_correction :: Union{Tuple{Function, Function}, Nothing} = nothing,
            ac_regularisation :: Bool = true, 
            evaluate_ac_spectrum ::Bool = true, 
            inspect :: Function = (args...) -> nothing,
            loggingstream :: IO = devnull,
            kwargs... ) where {TR<:Number, Tg<:Number}
    
    if any(g .<= 0) # limit unfolding to non-zero bins
        nonzero = g .> 0
        @warn "Limiting p_RUN to $(sum(nonzero)) of $(length(g)) observeable non-zero bins"
        g = g[nonzero]
        R = R[nonzero, :]
    end
    
    # check arguments
    m = size(R, 2) # dimension of f
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    end
    if m > size(R, 1)
        @warn "p_RUN is performed on more target than observable m - results may be unsatisfactory"
    end

     # set up acceptance correction
    if acceptance_correction !== nothing
        ac, inv_ac = acceptance_correction
        if ac_regularisation
          a = inv_ac(ones(m))
        else
          a = nothing
        end
    else
        if evaluate_ac_spectrum
          @warn "Spectrum cannot be acceptance corrected because acceptance_correction object is not given"
          evaluate_ac_spectrum = false
        elseif ac_regularisation
          @warn "Performing acceptance correction regularisation requires a given acceptance_correction object"
          ac_regularisation = false
        end
        a = nothing
    end
    
    # set up regularized loss function
    C = _tikhonov_binning(m)
    l = _maxl_l(R,g) 
    C_l = _C_l(tau,C; a)
    l_reg = f -> l(f) + C_l(f)
    
    # regularized gradient
    g!(G, x) = begin     
      grad_l = _maxl_g(R, g)(x)
      grad_C = _C_g(tau, C; a)(x)
      grad_reg = grad_l .+ grad_C
      for j=1:12 
        G[j] = grad_reg[j]
      end
    end

    # regularized Hessian
    h!(H, x) = begin 
      hess_l = _maxl_H(R,g)(x)
      hess_C = _C_H(tau, C; a)(x)
      hess_reg = hess_l .+ hess_C
      J,K = size(H)
      for j=1:J, k=1:K
          H[j,k] = hess_reg[j,k]
      end
    end

    # performs interior-point Newton optimization in bounds of [0.0, Inf]
    lx = fill(0.0, m); ux = fill(Inf, m)
    dfc = TwiceDifferentiableConstraints(lx, ux)
    df = TwiceDifferentiable(l_reg, g!, h!, x0)

    conf = Optim.Options(g_tol = epsilon,
                        iterations = K, # max iterations
                        allow_f_increases = true,
                        store_trace = true,
                        extended_trace = true,
                        successive_f_tol = 2)

    res = optimize(df, dfc, x0, IPNewton(), conf)

    # evaluation
    epsilon = Optim.g_norm_trace(res)
    f = Optim.x_trace(res)
    k = Optim.iterations(res)
    
    if evaluate_ac_spectrum
      inspect.(ac.(f), collect(0:k), epsilon)
      return ac(f[k])
    else
      inspect.(f, collect(0:k), epsilon)
      return f[k]
    end
 
  end
