using ScikitLearn, Distances
using CherenkovDeconvolution.Util
using CherenkovDeconvolution.Sklearn
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end


# utilities
include("util.jl")
include("sklearn.jl")

# methods
include("dsea.jl")
include("run.jl")
include("ibu.jl")

