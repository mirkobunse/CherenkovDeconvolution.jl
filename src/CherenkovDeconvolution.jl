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

# utility modules
include("DeconvUtil.jl")
include("Binnings.jl")
include("Stepsizes.jl")
export DeconvUtil
using .Stepsizes
export ConstantStepsize, DEFAULT_STEPSIZE, ExpDecayStepsize, MulDecayStepsize, Stepsize, stepsize

# deconvolution methods
include("Methods.jl")
using .Methods
using .Methods: run # solve the conflict with Base.run
export dsea, ibu, p_run, run, svd # re-export

# optimized stepsizes
include("OptimizedStepsizes.jl")
using .OptimizedStepsizes
export LsqStepsize, OptimizedStepsize, RunStepsize

end # module
