using ScikitLearn, Distances
using CherenkovDeconvolution.Util
using CherenkovDeconvolution.Sklearn
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

srand(42) # make tests reproducible

# utilities
include("util.jl")
include("sklearn.jl")

# methods
include("methods/dsea.jl")
include("methods/run.jl")
include("methods/ibu.jl")

