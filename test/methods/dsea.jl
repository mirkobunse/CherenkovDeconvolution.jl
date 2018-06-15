# 
# Unit tests for dsea
# 

@testset "_dsea_weights" begin
    for _ in 1:10
        # random array
        num_bins  = rand(1:100)
        num_items = rand(1:1000)
        y_train = rand(1:num_bins, num_items)
        w_bin   = rand(num_bins)
        
        w_train = CherenkovDeconvolution._dsea_weights(y_train, w_bin)
        
        # consider Laplace correction (correct weights pass the test without assertion)
        unequal = w_train .!= w_bin[y_train] # indices which have to be checked
        if any(unequal)
            info("$(sum(unequal)) weights are not equal - checking Laplace correction")
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
        f_plus, alpha_out = CherenkovDeconvolution._dsea_step(k_dummy, f, f_prev, alpha_const)
        @test alpha_const == alpha_out
        @test all( f_plus .== f_prev + (f - f_prev) * alpha_const )
    end
end

