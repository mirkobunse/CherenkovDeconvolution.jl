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
"""
    module OptimizedStepsizes

This module contains a collection of stepsize strategies that are optimized with respect
to different criteria.
"""
module OptimizedStepsizes

using LinearAlgebra, Optim
using ..DeconvUtil, ..Methods, ..Stepsizes

export LsqStepsize, OptimizedStepsize, RunStepsize

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
function Stepsizes.stepsize(s::OptimizedStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64)
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
    C = Methods._tikhonov_binning(size(R, 2)) # regularization matrix (from run.jl)
    maxl_l = Methods._maxl_l(R, g)         # function of f (from run.jl)
    maxl_C = Methods._C_l(tau, C)          # regularization term (from run.jl)
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
    C = diagm(0 => ones(size(R, 2)))     # minimum-norm regularization matrix
    lsq_l = Methods._lsq_l(R, g)         # function of f (from run.jl)
    lsq_C = Methods._C_l(tau, C)         # regularization term (from run.jl)
    objective = f -> lsq_l(f) + lsq_C(f) # regularized objective function
    return OptimizedStepsize(objective, decay)
end

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

end # module
