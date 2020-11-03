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
    alpha_adaptive_run(x_data, x_train, y_train[, tau=0; bins_y, bins_x, warn=false, decay=false])

Return a `Function` object with the signature required by the `alpha` parameter in `dsea`.

This object adapts the DSEA step size to the current estimate by maximizing the likelihood
of the next estimate in the search direction of the current iteration.

Setting `decay=true` will enforce that a_k+1 <= a_k, i.e. the step sizes never increase.
"""
function alpha_adaptive_run( x_data  :: AbstractVector{T},
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
    
    # set up negative log likelihood function to be minimized
    C = _tikhonov_binning(size(R, 2))       # regularization matrix (from run.jl)
    maxl_l = _maxl_l(R, g)                  # function of f (from run.jl)
    maxl_C = _C_l(tau, C)                   # regularization term (from run.jl)
    negloglike = f -> maxl_l(f) + maxl_C(f) # regularized objective function
    
    # return step size function
    return (k::Int, pk::Vector{Float64}, f::Vector{Float64}, a_prev::Float64) -> begin
        a_min, a_max = _alpha_range(pk, f)
        if decay
            a_max = min(a_max, a_prev) # never increase the step size
        end
        if a_max > a_min
            optimize(a -> negloglike(f + a * pk), a_min, a_max).minimizer # from Optim.jl
        else
            min(a_min, a_prev) # only one value is feasible
        end
    end
end

"""
    alpha_adaptive_lsq(x_data, x_train, y_train[, tau=0; bins_y, bins_x, warn=false, decay=false])

Return a `Function` object with the signature required by the `alpha` parameter in `dsea`.

This object adapts the DSEA step size to the current estimate by solving a least squares
objective in the search direction of the current iteration.

Setting `decay=true` will enforce that a_k+1 <= a_k, i.e. the step sizes never increase.
"""
function alpha_adaptive_lsq( x_data  :: AbstractVector{T},
                             x_train :: AbstractVector{T},
                             y_train :: AbstractVector{T},
                             tau     :: Number = 0.0;
                             bins_y  :: AbstractVector{T} = 1:maximum(y_train),
                             bins_x  :: AbstractVector{T} = 1:maximum(vcat(x_data, x_train)),
                             warn    :: Bool = false,
                             decay   :: Bool = false ) where T<:Int
    # set up the discrete deconvolution problem
    R = DeconvUtil.normalizetransfer(DeconvUtil.fit_R(y_train, x_train, bins_y=bins_y, bins_x=bins_x, normalize=false), warn=warn)
    g = DeconvUtil.fit_pdf(x_data, bins_x, normalize=false) # absolute counts instead of pdf

    # set up negative log likelihood function to be minimized
    C = _tikhonov_binning(size(R, 2))     # regularization matrix (from run.jl)
    lsq_l = _lsq_l(R, g)                  # function of f (from run.jl)
    lsq_C = _C_l(tau, C)                  # regularization term (from run.jl)
    negloglike = f -> lsq_l(f) + lsq_C(f) # regularized objective function

    # return step size function
    return (k::Int, pk::Vector{Float64}, f::Vector{Float64}, a_prev::Float64) -> begin
        a_min, a_max = _alpha_range(pk, f)
        if decay
            a_max = min(a_max, a_prev) # never increase the step size
        end
        if a_max > a_min
            optimize(a -> negloglike(f + a * pk), a_min, a_max).minimizer # from Optim.jl
        else
            min(a_min, a_prev) # only one value is feasible
        end
    end
end

"""
    alpha_decay_exp(eta::Float64, a_1::Float64=1.0)

Return a `Function` object with the signature required by the `alpha` parameter in `dsea`.
This object reduces the `a_1` stepsize taken in iteration 1 by `eta` in each subsequent
iteration:

    alpha = a_1 * eta^(k-1).
"""
alpha_decay_exp(eta::Float64, a_1::Float64=1.0) =
    (k::Int, pk::Vector{Float64}, f::Vector{Float64}, a_prev::Float64) -> a_1 * eta^(k-1)

"""
    alpha_decay_mul(eta::Float64, a_1::Float64=1.0)

Return a `Function` object with the signature required by the `alpha` parameter in `dsea`.
This object reduces the `a_1` stepsize taken in iteration 1 by `eta` in each subsequent
iteration:

    alpha = a_1 * k ^ (eta-1)
    
For example, eta=.5 yields alpha = 1/sqrt(k).
"""
alpha_decay_mul(eta::Float64, a_1::Float64=1.0) =
    (k::Int, pk::Vector{Float64}, f::Vector{Float64}, a_prev::Float64) -> a_1 * k^(eta-1)

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
