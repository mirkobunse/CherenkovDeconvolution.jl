using ScikitLearn, Distances, Discretizers, NBInclude
using CherenkovDeconvolution
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

# 
# additional helpers
# 
@testset "recoding of labels" begin
    y = [1, 4, 3, 3, 4, 1, 3] # 2 is missing
    y_rec, recode_dict = CherenkovDeconvolution._recode_labels(y, 1:4)
    @test find(y .== 1) == find(y_rec .== 1)
    @test find(y .== 3) == find(y_rec .== 2)
    @test find(y .== 4) == find(y_rec .== 3)
    @test map(i -> recode_dict[i], y_rec) == y
    
    @test CherenkovDeconvolution._recode_result([.1, .2, .3], recode_dict) == [.1, 0, .2, .3]
    
    y[y .== 4] = 2 # 4 is missing, instead
    y_rec, recode_dict = CherenkovDeconvolution._recode_labels(y, 1:4)
    @test map(i -> recode_dict[i], y_rec) == y
    @test CherenkovDeconvolution._recode_result([.1, .2, .3], recode_dict) == [.1, .2, .3, 0]
    
    y = [1, 4, 4, 4, 4, 1, 4] # 2 and 3 is missing
    y_rec, recode_dict = CherenkovDeconvolution._recode_labels(y, 1:4)
    @test find(y .== 1) == find(y_rec .== 1)
    @test find(y .== 4) == find(y_rec .== 2)
    @test map(i -> recode_dict[i], y_rec) == y
end

# methods
include("methods/dsea.jl")
include("methods/run.jl")
include("methods/ibu.jl")

# check that no error occurs in example notebooks
nbinclude("../example/getting-started.ipynb")

