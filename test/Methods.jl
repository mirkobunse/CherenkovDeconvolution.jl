# 
# Unit tests for the Methods module
# 
@testset "CherenkovDeconvolution.Methods, as tested in test/Methods.jl" begin

    # test the recoding of labels
    y = [1, 4, 3, 3, 4, 1, 3] # 2 is missing
    recode_dict, y_rec = Methods._recode_indices(1:4, y)
    @test findall(y .== 1) == findall(y_rec .== 1)
    @test findall(y .== 3) == findall(y_rec .== 2)
    @test findall(y .== 4) == findall(y_rec .== 3)
    @test map(i -> recode_dict[i], y_rec) == y
    
    @test Methods._recode_result([.1, .2, .3], recode_dict) == [.1, 0, .2, .3]
    
    y[y .== 4] .= 2 # 4 is missing, instead
    recode_dict, y_rec = Methods._recode_indices(1:4, y)
    @test map(i -> recode_dict[i], y_rec) == y
    @test Methods._recode_result([.1, .2, .3], recode_dict) == [.1, .2, .3, 0]
    
    y = [1, 4, 4, 4, 4, 1, 4] # 2 and 3 is missing
    recode_dict, y_rec = Methods._recode_indices(1:4, y)
    @test findall(y .== 1) == findall(y_rec .== 1)
    @test findall(y .== 4) == findall(y_rec .== 2)
    @test map(i -> recode_dict[i], y_rec) == y
    
    x1 = ones(Int64, 4) * 2 # 4 times the 2
    x2 = ones(Int64, 3) * 4 # 3 times the 4
    recode_dict, x1_rec, x2_rec = Methods._recode_indices(1:4, x1, x2)
    @test x1_rec == ones(4)
    @test x2_rec == ones(3) .* 2
    @test map(i -> recode_dict[i], x1_rec) == x1
    @test map(i -> recode_dict[i], x2_rec) == x2

end # testset

# methods
include("methods/dsea.jl")
include("methods/svd.jl")
include("methods/run.jl")
include("methods/ibu.jl")
