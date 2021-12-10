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
using ..Binnings, ..DeconvUtil, ..Methods, ..Stepsizes

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
function Stepsizes.value(s::OptimizedStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64)
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

# the DiscreteStepsize implements the RunStepsize and LsqStepsize
abstract type DiscreteStepsizeObjective end
struct DiscreteStepsize{O<:DiscreteStepsizeObjective} <: Stepsize
    is_initialized :: Ref{Bool} # mutable field
    optimized_stepsize :: Ref{OptimizedStepsize}
    binning :: Binning
    decay :: Bool
    tau :: Float64
    warn :: Bool
end

Stepsizes.value(s::DiscreteStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    if s.is_initialized[]
        Stepsizes.value(s.optimized_stepsize[], k, p, f, a)
    else
        throw(ArgumentError("The stepsize is not yet initialized"))
    end

function Stepsizes.initialize!(
        s::DiscreteStepsize,
        X_obs::AbstractArray{T,N},
        X_trn::AbstractArray{T,N},
        y_trn::AbstractVector{I}
        ) where {T,N,I<:Integer}
    # discretize the problem statement into a system of linear equations
    d = BinningDiscretizer(s.binning, X_trn, y_trn) # fit the binning strategy with labeled data
    x_obs = encode(d, X_obs) # apply it to the feature vectors
    x_trn = encode(d, X_trn)
    R = DeconvUtil.normalizetransfer(DeconvUtil.fit_R(y_trn, x_trn; bins_x=bins(d), normalize=false); warn=s.warn)
    g = DeconvUtil.fit_pdf(x_obs, bins(d); normalize=false) # absolute counts instead of pdf

    # update the stepsize
    s.optimized_stepsize[] = OptimizedStepsize(objective(s, R, g), s.decay)
    s.is_initialized[] = true
    return s
end

struct RunObjective <: DiscreteStepsizeObjective end
struct LsqObjective <: DiscreteStepsizeObjective end
const RunStepsize = DiscreteStepsize{RunObjective}
const LsqStepsize = DiscreteStepsize{LsqObjective}

"""
    RunStepsize(binning; kwargs...)

Adapt the step size by maximizing the likelihood of the next estimate in the search direction
of the current iteration, much like in the `RUN` deconvolution method.

**Keyword arguments:**

- `decay = false`
  specifies whether `a_k+1 <= a_k` is enforced so that step sizes never increase.
- `tau = 0.0`
  determines the regularisation strength.
- `warn = false`
  specifies whether warnings should be emitted for debugging purposes.
"""
RunStepsize(binning::Binning; decay::Bool=false, tau::Float64=0.0, warn::Bool=false) =
    RunStepsize(Ref{Bool}(false), Ref{OptimizedStepsize}(), binning, decay, tau, warn)

# set up the negative log likelihood function to be minimized
function objective(s::RunStepsize, R::Matrix{Float64}, g::Vector{Int})
    C = Methods._tikhonov_binning(size(R, 2)) # regularization matrix (from run.jl)
    maxl_l = Methods._maxl_l(R, g)    # function of f (from run.jl)
    maxl_C = Methods._C_l(s.tau, C)   # regularization term (from run.jl)
    return f -> maxl_l(f) + maxl_C(f) # regularized objective function
end

"""
    LsqStepsize(binning; kwargs...)

Adapt the step size by solving a least squares objective in the search direction of the
current iteration.

**Keyword arguments:**

- `decay = false`
  specifies whether `a_k+1 <= a_k` is enforced so that step sizes never increase.
- `tau = 0.0`
  determines the regularisation strength.
- `warn = false`
  specifies whether warnings should be emitted for debugging purposes.
"""
LsqStepsize(binning::Binning; decay::Bool=false, tau::Float64=0.0, warn::Bool=false) =
    LsqStepsize(Ref{Bool}(false), Ref{OptimizedStepsize}(), binning, decay, tau, warn)

# set up the negative least-squares function to be minimized
function objective(s::LsqStepsize, R::Matrix{Float64}, g::Vector{Int})
    C = diagm(0 => ones(size(R, 2))) # minimum-norm regularization matrix
    lsq_l = Methods._lsq_l(R, g)     # function of f (from run.jl)
    lsq_C = Methods._C_l(s.tau, C)   # regularization term (from run.jl)
    return f -> lsq_l(f) + lsq_C(f)  # regularized objective function
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
    a_max = any(a_zero .>= 0) ? minimum(a_zero[a_zero .>= 0]) : 0.
    return a_min, a_max
end

# deprecated syntax
struct BufferBinning <: Binning
    buffer :: Vector{Vector{Int}}
    bins :: Vector{Int}
end
struct BufferDiscretizer <: BinningDiscretizer
    buffer :: Vector{Vector{Int}}
    bins :: Vector{Int}
end
Binnings.BinningDiscretizer(b::BufferBinning, X_trn, y_trn) =
    BufferDiscretizer(b.buffer, b.bins) # copy
Binnings.encode(d::BufferDiscretizer, X_obs) =
    if size(X_obs, 1) == length(d.buffer[1])
        return d.buffer[1]
    elseif size(X_obs, 1) == length(d.buffer[2])
        return d.buffer[2]
    else
        error("Size does not match")
    end
Binnings.bins(d::BufferDiscretizer) = d.bins
function RunStepsize(
        x_obs  :: AbstractVector{T},
        x_trn  :: AbstractVector{T},
        y_trn  :: AbstractVector{T},
        tau    :: Number = 0.0;
        bins_y :: AbstractVector{T} = 1:maximum(y_trn),
        bins_x :: AbstractVector{T} = 1:maximum(vcat(x_obs, x_trn)),
        warn   :: Bool = false,
        decay  :: Bool = false ) where T<:Int
    Base.depwarn(join([
        "`RunStepsize(data, config)` is deprecated; ",
        "please call `initialize!(RunStepsize(config), data)` instead"
    ]), :RunStepsize)
    if length(x_obs) == length(x_trn)
        @warn "oh-oh"
    end
    s = RunStepsize(Ref{Bool}(false), Ref{OptimizedStepsize}(), BufferBinning([x_obs, x_trn], bins_x), decay, tau, warn)
    return initialize!(s, reshape(x_obs, (length(x_obs), 1)), reshape(x_trn, (length(x_trn), 1)), y_trn)
end
function LsqStepsize(
        x_obs  :: AbstractVector{T},
        x_trn  :: AbstractVector{T},
        y_trn  :: AbstractVector{T},
        tau    :: Number = 0.0;
        bins_y :: AbstractVector{T} = 1:maximum(y_trn),
        bins_x :: AbstractVector{T} = 1:maximum(vcat(x_obs, x_trn)),
        warn   :: Bool = false,
        decay  :: Bool = false ) where T<:Int
    Base.depwarn(join([
        "`LsqStepsize(data, config)` is deprecated; ",
        "please call `initialize!(LsqStepsize(config), data)` instead"
    ]), :LsqStepsize)
    if length(x_obs) == length(x_trn)
        @warn "oh-oh"
    end
    s = LsqStepsize(Ref{Bool}(false), Ref{OptimizedStepsize}(), BufferBinning([x_obs, x_trn], bins_x), decay, tau, warn)
    return initialize!(s, reshape(x_obs, (length(x_obs), 1)), reshape(x_trn, (length(x_trn), 1)), y_trn)
end

end # module
