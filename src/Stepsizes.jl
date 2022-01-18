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
    module Stepsizes

This module contains a collection of basic stepsize strategies for deconvolution methods.

**See also:** `module OptimizedStepsizes`
"""
module Stepsizes

using LinearAlgebra, Optim

export
    ConstantStepsize,
    DEFAULT_STEPSIZE,
    ExpDecayStepsize,
    initialize_prefit!,
    initialize_deconvolve!,
    MulDecayStepsize,
    Stepsize,
    value

"""
    abstract type stepsize end

Abstract supertype for step sizes in deconvolution.

**See also:** `stepsize`.
"""
abstract type Stepsize end

"""
    value(s, k, p, f, a)

Use the `Stepsize` object `s` to compute a step size for iteration number `k` with
the search direction `p`, the previous estimate `f`, and the previous step size `a`.

**See also:** `ConstantStepsize`, `RunStepsize`, `LsqStepsize`, `ExpDecayStepsize`,
`MulDecayStepsize`.
"""
value(s::Stepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    throw(ArgumentError("Not implemented for type $(typeof(s))"))

"""
    initialize_prefit!(s, X_trn, y_trn)

Prepare the stepsize strategy `s` with the training set `(X_trn, y_trn)`.

**See also:** `initialize_deconvolve!`.
"""
initialize_prefit!(s::Stepsize, X_trn::Any, y_trn::AbstractVector{I}) where {I<:Integer} = s

"""
    initialize_deconvolve!(s, X_obs)

Prepare the stepsize strategy `s` with the observed features in `X_obs`.

**See also:** `initialize_prefit!`.
"""
initialize_deconvolve!(s::Stepsize, X_obs::Any) = s # there's nothing to prepare by default

"""
    ConstantStepsize(alpha)

Choose the constant step size `alpha` in every iteration.
"""
struct ConstantStepsize <: Stepsize
    alpha::Float64
end
value(s::ConstantStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    s.alpha

"""
    ExpDecayStepsize(eta, a=1.0)

Reduce the first stepsize `a` by `eta` in each iteration:

    value(ExpDecayStepsize(eta, a), k, ...) == a * eta^(k-1)
"""
struct ExpDecayStepsize <: Stepsize
    eta::Float64
    a::Float64
    ExpDecayStepsize(eta::Float64, a::Float64=1.0) = new(eta, a)
end
value(s::ExpDecayStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    s.a * s.eta^(k-1)

"""
    MulDecayStepsize(eta, a=1.0)

Reduce the first stepsize `a` by `eta` in each iteration:

    value(MulDecayStepsize(eta, a), k, ...) == a * k^(eta-1)
"""
struct MulDecayStepsize <: Stepsize
    eta::Float64
    a::Float64
    MulDecayStepsize(eta::Float64, a::Float64=1.0) = new(eta, a)
end
value(s::MulDecayStepsize, k::Int, p::Vector{Float64}, f::Vector{Float64}, a::Float64) =
    s.a * k^(s.eta-1)

"""
    const DEFAULT_STEPSIZE = ConstantStepsize(1.0)

The default stepsize in all deconvolution methods.
"""
const DEFAULT_STEPSIZE = ConstantStepsize(1.0)

end # module
