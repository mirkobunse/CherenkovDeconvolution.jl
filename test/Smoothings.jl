# 
# Unit tests for the Util module
# 
@testset "CherenkovDeconvolution.Smoothings, as tested in test/Smoothings.jl" begin

for i in 1:10
    num_bins = rand(100:1000)
    f_rand = DeconvUtil.normalizepdf(rand(num_bins))

    # no smoothing does not change the input
    @test f_rand == apply(NoSmoothing(), f_rand)

    # simple order 1 check
    f_smooth = apply(PolynomialSmoothing(1; warn=false), f_rand) # apply smoothing of order 1 to f_rand
    diffs = f_smooth[2:end] - f_smooth[1:end-1] # array of finite differences
    @test all(isapprox.(diffs, mean(diffs))) # all differences are approximately equal

    # multiple smoothings return approximately same array
    smoothing = PolynomialSmoothing(i; warn=false)
    f_smooth = apply(smoothing, f_rand)
    @test isapprox(f_smooth, apply(smoothing, f_smooth), rtol=0.01)

end

end # end of util testset