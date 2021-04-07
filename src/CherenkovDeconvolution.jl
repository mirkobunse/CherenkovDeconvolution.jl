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
module CherenkovDeconvolution

using Reexport
export ConstantStepsize, DEFAULT_STEPSIZE, Stepsize, stepsize # abstract type and API must be defined here

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
const DEFAULT_STEPSIZE = ConstantStepsize(1.0)

# utility modules
include("DeconvUtil.jl")
include("Binnings.jl")
export DeconvUtil

# deconvolution methods
include("Methods.jl")
using .Methods
using .Methods: run # solve the conflict with Base.run
export dsea, ibu, p_run, run, svd # re-export

include("Stepsizes.jl")
using .Stepsizes
export ConstantStepsize, RunStepsize, LsqStepsize, ExpDecayStepsize, MulDecayStepsize, DEFAULT_STEPSIZE

end # module
