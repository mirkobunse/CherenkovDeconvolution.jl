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

include("DeconvUtil.jl") # utility functions
export DeconvUtil # export as a module; do not re-export members

include("Binnings.jl")
@reexport using .Binnings

include("Stepsizes.jl")
@reexport using .Stepsizes

include("Methods.jl")
@reexport using .Methods
using .Methods: run # solve the naming conflict with Base.run
export run

include("OptimizedStepsizes.jl")
@reexport using .OptimizedStepsizes

# redirect deprecated methods
@deprecate alpha_decay_exp ExpDecayStepsize
@deprecate alpha_decay_mul MulDecayStepsize
@deprecate alpha_adaptive_run RunStepsize
@deprecate alpha_adaptive_lsq LsqStepsize

end # module
