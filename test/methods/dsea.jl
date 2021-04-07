# 
# Unit tests for dsea
# 
@testset "Methods related to DSEA, as tested in test/methods/dsea.jl" begin


@testset "_dsea_weights" begin
    for _ in 1:10
        # random array
        num_bins  = rand(1:100)
        num_items = rand(1:1000)
        y_train = rand(1:num_bins, num_items)
        w_bin   = rand(num_bins)
        
        w_train = Methods._dsea_weights(y_train, w_bin)
        
        # consider Laplace correction (correct weights pass the test without assertion)
        unequal = w_train .!= w_bin[y_train] # indices which have to be checked
        if any(unequal)
            @info "$(sum(unequal)) weights are not equal - checking Laplace correction"
            @test all(w_train[unequal] .== 1/length(y_train))
            @test all(w_bin[y_train[unequal]] .<= 1/length(y_train))
        end # else, all weights are equal -> pass test
    end
end

@testset "_dsea_step" begin
    for _ in 1:10
        num_bins = rand(1:100)
        k_dummy  = rand(1:100)
        f        = rand(num_bins)
        f_prev   = rand(num_bins)
        alpha_const = 4 * (rand() - .5) # in [-2, 2)
        
        # test constant step size
        f_plus, alpha_out = Methods._dsea_step(k_dummy, f, f_prev, Inf, ConstantStepsize(alpha_const))
        @test alpha_const == alpha_out
        @test all( f_plus .== f_prev + (f - f_prev) * alpha_const )
    end
end

@testset "_alpha_range" begin
    
    PRECISION = 1e-4
    
    # new implementation of _alpha_range should equal this correct brute-force version
    function _alpha_range_old(pk::Array{Float64,1}, f::Array{Float64,1})
        alphas     = 0:PRECISION:1# all alphas
        admissible = map(a -> all(f + a * pk .>= 0), alphas)
        alpha_min  = alphas[findfirst(admissible)]
        alpha_max  = alphas[ findlast(admissible)]
        return alpha_min, alpha_max
    end
    
    for _ in 1:10
        num_bins = rand(1:100)
        f        = rand(num_bins)
        a_max    = rand()      # maximum alpha value
        pk       = -f ./ a_max # f + a_max * pk == 0 (approximately)
        
        # find range of admissible alphas
        range_old = _alpha_range_old(pk, f)
        range_new = Stepsizes._alpha_range(pk, f)
        
        # old method is only approximate, so the new result has to be rounded for comparison
        range_new_rounded = floor.(range_new, digits=abs(convert(Int, log10(PRECISION))))
        @test range_new_rounded[1] == range_old[1]
        @test range_new_rounded[2] == range_old[2]
    end
    
end


end