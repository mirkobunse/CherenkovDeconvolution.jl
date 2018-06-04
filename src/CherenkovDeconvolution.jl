# 
# CherenkovDeconvolution.jl
# Copyright 2018 Mirko Bunse
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


using DataFrames

export Util, Sklearn
export dsea


# utility module
module Util
    include("util.jl")
end

# optional sklearn utilities
module Sklearn
    using Requires
    @require ScikitLearn begin
        info("Utilities of ScikitLearn.jl are available in CherenkovDeconvolution.Sklearn")
        include("sklearn.jl")
    end
end


# deconvolution methods
# include("methods/run.jl")
# include("methods/ibu.jl")
include("methods/dsea.jl")


end # module
