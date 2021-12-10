using
    CherenkovDeconvolution,
    Discretizers,
    Distances,
    PyCall,
    Random,
    ScikitLearn,
    SparseArrays,
    Statistics,
    Test

# support for sparse Python matrices
SCIPY_SPARSE = pyimport_conda("scipy.sparse", "scipy")
SparseArrays.SparseMatrixCSC(x::PyObject) =
    if pyisinstance(x, SCIPY_SPARSE.csc.csc_matrix)
        return SparseArrays.SparseMatrixCSC( # TODO replace .+ 1 (requires a copy)
            x.shape[1],     # m
            x.shape[2],     # n
            x.indptr .+ 1,  # colptr
            x.indices .+ 1, # rowval
            x.data          # nzval
        ) # see https://docs.julialang.org/en/v1/stdlib/SparseArrays/#man-csc
    else
        throw(ArgumentError(
            "Cannot `convert` a PyObject of type `$(x.__class__.__name__)` to an object of type SparseMatrixCSC"
        ))
    end

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
@scoped_nbinclude "../docs/01-getting-started.ipynb"
@scoped_nbinclude "../docs/02-inspection.ipynb"
@scoped_nbinclude "../docs/03-adaptive-stepsize.ipynb"

# check that deprecation redirections work
@scoped_nbinclude "deprecated/01-getting-started.ipynb"
@scoped_nbinclude "deprecated/02-inspection.ipynb"
@scoped_nbinclude "deprecated/03-adaptive-stepsize.ipynb"
