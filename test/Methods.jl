# 
# Unit tests for the Methods module
# 
@testset "Methods.LabelSanitizer, as tested in test/Methods.jl" begin

    # test the recoding of labels
    y = [1, 4, 3, 3, 4, 1, 3] # 2 is missing
    s = LabelSanitizer(y)
    @test s.bins == [1, 3, 4]
    @test encode_labels(s, y) == [1, 3, 2, 2, 3, 1, 2]
    @test encode_prior(s, [.1, .2, .3, .4]) == [.1, .3, .4]
    @test decode_estimate(s, [.1, .2, .3]) == [.1, 0, .2, .3]
    
    y[y .== 4] .= 2 # 4 is missing this time
    s = LabelSanitizer(y, 4) # we must specify the number of bins now
    @test s.bins == [1, 2, 3]
    @test encode_labels(s, y) == y # no recoding happens
    @test encode_prior(s, [.1, .2, .3, .4]) == [.1, .2, .3]
    @test decode_estimate(s, [.1, .2, .3]) == [.1, .2, .3, 0]
    
    y = [1, 4, 4, 4, 4, 1, 4] # 2 and 3 are missing
    s = LabelSanitizer(y)
    @test s.bins == [1, 4]
    @test encode_labels(s, y) == [1, 2, 2, 2, 2, 1, 2]
    @test encode_prior(s, [.1, .2, .3, .4]) == [.1, .4]
    @test decode_estimate(s, [.1, .2]) == [.1, 0, 0, .2]
    
    y1 = ones(Int64, 4) * 2 # 4 times the 2
    y2 = ones(Int64, 3) * 4 # 3 times the 4
    s = LabelSanitizer(vcat(y1, y2), 4)
    @test encode_labels(s, y1) == ones(4)
    @test encode_labels(s, y2) == ones(3) .* 2
    @test encode_prior(s, [.1, .2, .3, .4]) == [.2, .4]
    @test decode_estimate(s, [.1, .2]) == [0, .1, 0, .2]

end # testset

# methods
include("methods/dsea.jl")
include("methods/svd.jl")
include("methods/run.jl")
include("methods/ibu.jl")
