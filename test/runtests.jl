using Test, Random, Statistics
using ScikitLearn, Distances, Discretizers, NBInclude
using CherenkovDeconvolution

Random.seed!(42) # make tests reproducible

include("DeconvUtil.jl")
include("Binnings.jl")
include("Methods.jl")

# check that no error occurs in example notebooks
@nbinclude "../doc/01-getting-started.ipynb"
@nbinclude "../doc/02-inspection.ipynb"
@nbinclude "../doc/03-adaptive-stepsize.ipynb"

# check that deprecation redirections work
@nbinclude "deprecated/01-getting-started.ipynb"
@nbinclude "deprecated/02-inspection.ipynb"
@nbinclude "deprecated/03-adaptive-stepsize.ipynb"
