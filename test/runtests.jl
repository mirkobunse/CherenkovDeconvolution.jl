using Test, Random, Statistics
using ScikitLearn, Distances, Discretizers
using CherenkovDeconvolution

Random.seed!(42) # make tests reproducible

include("DeconvUtil.jl")
include("Binnings.jl")
include("Methods.jl")

# execute each jupyter notebook in its own module
macro scoped_nbinclude(path::AbstractString)
    return quote
        @eval module $(Symbol(replace(path, r"[./\-\d]" => "")))
            using NBInclude
            @nbinclude $path
        end
    end
end

# check that no error occurs in example notebooks
@scoped_nbinclude "../doc/01-getting-started.ipynb"
@scoped_nbinclude "../doc/02-inspection.ipynb"
@scoped_nbinclude "../doc/03-adaptive-stepsize.ipynb"

# check that deprecation redirections work
@scoped_nbinclude "deprecated/01-getting-started.ipynb"
@scoped_nbinclude "deprecated/02-inspection.ipynb"
@scoped_nbinclude "deprecated/03-adaptive-stepsize.ipynb"
