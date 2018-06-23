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
    recode_dict, y_rec = CherenkovDeconvolution._recode_indices(1:4, y)
    @test find(y .== 1) == find(y_rec .== 1)
    @test find(y .== 3) == find(y_rec .== 2)
    @test find(y .== 4) == find(y_rec .== 3)
    @test map(i -> recode_dict[i], y_rec) == y
    
    @test CherenkovDeconvolution._recode_result([.1, .2, .3], recode_dict) == [.1, 0, .2, .3]
    
    y[y .== 4] = 2 # 4 is missing, instead
    recode_dict, y_rec = CherenkovDeconvolution._recode_indices(1:4, y)
    @test map(i -> recode_dict[i], y_rec) == y
    @test CherenkovDeconvolution._recode_result([.1, .2, .3], recode_dict) == [.1, .2, .3, 0]
    
    y = [1, 4, 4, 4, 4, 1, 4] # 2 and 3 is missing
    recode_dict, y_rec = CherenkovDeconvolution._recode_indices(1:4, y)
    @test find(y .== 1) == find(y_rec .== 1)
    @test find(y .== 4) == find(y_rec .== 2)
    @test map(i -> recode_dict[i], y_rec) == y
    
    x1 = ones(Int64, 4) * 2 # 4 times the 2
    x2 = ones(Int64, 3) * 4 # 3 times the 4
    recode_dict, x1_rec, x2_rec = CherenkovDeconvolution._recode_indices(1:4, x1, x2)
    @test x1_rec == ones(4)
    @test x2_rec == ones(3) .* 2
    @test map(i -> recode_dict[i], x1_rec) == x1
    @test map(i -> recode_dict[i], x2_rec) == x2
end

# methods
include("methods/dsea.jl")
include("methods/run.jl")
include("methods/ibu.jl")

# check that no error occurs in example notebooks
nbinclude("../example/getting-started.ipynb")

