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
export Stepsize, stepsize, ConstantStepsize, RunStepsize, LsqStepsize, ExpDecayStepsize, MulDecayStepsize, DEFAULT_STEPSIZE

@deprecate alpha_adaptive_run RunStepsize
@deprecate alpha_adaptive_lsq LsqStepsize
@deprecate alpha_decay_exp ExpDecayStepsize
@deprecate alpha_decay_mul MulDecayStepsize

"""
    abstract type stepsize end

Abstract supertype for step sizes in deconvolution.

**See also:** `stepsize`.
"""
abstract type Stepsize end

"""
    stepsize(s, k, p, f, a)

Use the `Stepsize` object `s` to compute a step size for iteration number `k` with
the search direction `p`, the previous estimate `f`, and the previous step size `a`.

**See also:** `ConstantStepsize`, `RunStepsize`, `LsqStepsize`, `ExpDecayStepsize`,
`MulDecayStepsize`.
"""
stepsize(s::Stepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    error("Not implemented")

"""
    ConstantStepsize(alpha)

Choose the constant step size `alpha` in every iteration.
"""
struct ConstantStepsize <: Stepsize
    alpha::Float64
end
stepsize(s::ConstantStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    s.alpha

"""
    OptimizedStepsize(objective, decay)

A step size that is optimized over an `objective` function. If `decay=true`, then
the step sizes never increase.

**See also:** `RunStepsize`, `LsqStepsize`.
"""
struct OptimizedStepsize <: Stepsize
    objective::Function
    decay::Bool
end
function stepsize(s::OptimizedStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64)
    a_min, a_max = _alpha_range(p, f)
    if s.decay
        a_max = min(a_max, a) # never increase the step size
    end
    return if a_max > a_min
        optimize(x -> s.objective(f + x * p), a_min, a_max).minimizer # from Optim.jl
    else
        min(a_min, a) # only one value is feasible
    end
end

"""
    RunStepsize(x_data, x_train, y_train[, tau=0; bins_y, bins_x, warn=false, decay=false])

Adapt the step size by maximizing the likelihood of the next estimate in the search direction
of the current iteration.

The arguments of this function reflect a discretized deconvolution problem, as used in RUN.
Setting `decay=true` will enforce that a_k+1 <= a_k, i.e. the step sizes never increase.

**See also:** `OptimizedStepsize`.
"""
function RunStepsize( x_data  :: AbstractVector{T},
                      x_train :: AbstractVector{T},
                      y_train :: AbstractVector{T},
                      tau     :: Number = 0.0;
                      bins_y  :: AbstractVector{T} = 1:maximum(y_train),
                      bins_x  :: AbstractVector{T} = 1:maximum(vcat(x_data, x_train)),
                      warn    :: Bool = false,
                      decay   :: Bool = false ) where T<:Int
    # set up the discrete deconvolution problem
    R = DeconvUtil.normalizetransfer(DeconvUtil.fit_R(y_train, x_train, bins_y=bins_y, bins_x=bins_x, normalize=false), warn=warn)
    g = DeconvUtil.fit_pdf(x_data, bins_x, normalize = false) # absolute counts instead of pdf

    # set up the negative log likelihood function to be minimized
    C = _tikhonov_binning(size(R, 2))      # regularization matrix (from run.jl)
    maxl_l = _maxl_l(R, g)                 # function of f (from run.jl)
    maxl_C = _C_l(tau, C)                  # regularization term (from run.jl)
    objective = f -> maxl_l(f) + maxl_C(f) # regularized objective function
    return OptimizedStepsize(objective, decay)
end

"""
    LsqStepsize(x_data, x_train, y_train[, tau=0; bins_y, bins_x, warn=false, decay=false])

Adapt the step size by solving a least squares objective in the search direction of the
current iteration.

The arguments of this function reflect a discretized deconvolution problem, as used in RUN.
Setting `decay=true` will enforce that a_k+1 <= a_k, i.e. the step sizes never increase.

**See also:** `OptimizedStepsize`.
"""
function LsqStepsize( x_data  :: AbstractVector{T},
                      x_train :: AbstractVector{T},
                      y_train :: AbstractVector{T},
                      tau     :: Number = 0.0;
                      bins_y  :: AbstractVector{T} = 1:maximum(y_train),
                      bins_x  :: AbstractVector{T} = 1:maximum(vcat(x_data, x_train)),
                      warn    :: Bool = false,
                      decay   :: Bool = false ) where T<:Int
    # set up the discrete deconvolution problem
    R = DeconvUtil.normalizetransfer(DeconvUtil.fit_R(y_train, x_train, bins_y=bins_y, bins_x=bins_x, normalize=false), warn=warn)
    g = DeconvUtil.fit_pdf(x_data, bins_x, normalize = false) # absolute counts instead of pdf

    # set up the negative log likelihood function to be minimized
    C = _tikhonov_binning(size(R, 2))    # regularization matrix (from run.jl)
    lsq_l = _lsq_l(R, g)                 # function of f (from run.jl)
    lsq_C = _C_l(tau, C)                 # regularization term (from run.jl)
    objective = f -> lsq_l(f) + lsq_C(f) # regularized objective function
    return OptimizedStepsize(objective, decay)
end

"""
    ExpDecayStepsize(eta, a=1.0)

Reduce the first stepsize `a` by `eta` in each iteration:

    stepsize(ExpDecayStepsize(eta, a), k, ...) == a * eta^(k-1)
"""
struct ExpDecayStepsize <: Stepsize
    eta::Float64
    a::Float64
    ExpDecayStepsize(eta::Float64, a::Float64=1.0) = new(eta, a)
end
stepsize(s::ExpDecayStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    s.a * s.eta^(k-1)

"""
    MulDecayStepsize(eta, a=1.0)

Reduce the first stepsize `a` by `eta` in each iteration:

    stepsize(MulDecayStepsize(eta, a), k, ...) == a * k^(eta-1)
"""
struct MulDecayStepsize <: Stepsize
    eta::Float64
    a::Float64
    MulDecayStepsize(eta::Float64, a::Float64=1.0) = new(eta, a)
end
stepsize(s::MulDecayStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    s.a * k^(s.eta-1)

# range of admissible alpha values
function _alpha_range(pk::Vector{Float64}, f::Vector{Float64})
    if all(pk .== 0)
        return 0., 0.
    end # no reasonable direction

    # find alpha values for which the next estimate would be zero in one dimension
    a_zero = - (f[pk.!=0] ./ pk[pk.!=0]) # ignore zeros in pk, for which alpha is arbitrary

    # for positive pk[i] (negative a_zero[i]), alpha has to be larger than a_zero[i]
    # for negative pk[i] (positive a_zero[i]), alpha has to be smaller than a_zero[i]
    a_min = maximum(vcat(a_zero[a_zero .< 0], 0)) # vcat selects a_min = 0 if no pk[i]>0 is present
    a_max = minimum(a_zero[a_zero .>= 0])
    return a_min, a_max
end

const DEFAULT_STEPSIZE = ConstantStepsize(1.0)
