# 
# Unit tests for the Util module
# 
@testset "CherenkovDeconvolution.DeconvUtil, as tested in test/DeconvUtil.jl" begin


# dummy data
y = [2, 1, 3, 3, 3, 1]
Rdict = Dict(1 => 1,
             2 => 1,
             3 => 2,
             4 => 2)
x = get.(Ref(Rdict), y, -1)
R = zeros(Float64, (2, 3))
for i in 1:length(y)
    R[x[i], y[i]] += 1
end
R = DeconvUtil.normalizetransfer(R)


# normalizepdf
@testset "Normalize pdfs" begin
    for _ in 1:10
        num_bins  = rand(1:100)
        num_items = rand(1:1000)
        @test sum(DeconvUtil.normalizepdf(num_items .* rand(num_bins))) ≈ 1  atol=1e-6
    end
end


# pdfs
@test DeconvUtil.fit_pdf(y) == [ 1/3, 1/6, 1/2 ]

@testset "Random pdfs" begin
    for _ in 1:10
        
        # random array
        num_bins  = rand(1:100)
        num_items = rand(1:1000)
        rand_arr = rand(1:num_bins, num_items)
        
        # normalized pdf
        rand_pdf = DeconvUtil.fit_pdf(rand_arr, 1:num_bins)
        @test sum(rand_pdf) ≈ 1  atol=1e-6
        @test length(rand_pdf) == num_bins
        
        # unnormalized (histogram)
        rand_hist = DeconvUtil.fit_pdf(rand_arr, 1:num_bins, normalize = false)
        @test sum(rand_hist) == num_items
        @test length(rand_hist) == num_bins
        
        # laplace correction
        rand_arr[rand_arr .== 1] .= 2 # replace full class, which consequently gets probability zero
        @test DeconvUtil.fit_pdf(rand_arr, 1:num_bins, normalize = false)[1] == 0
        @test DeconvUtil.fit_pdf(rand_arr, 1:num_bins, laplace = true, normalize = false)[1] == 1
    end
end


# fit_R
@test size(DeconvUtil.fit_R(y, x)) == (2, 3)
@test DeconvUtil.fit_R(y, x) == R

@testset "Random transfer matrices" begin
    for _ in 1:10
        bins_y = 1:rand(1:100)
        bins_x = 1:rand(1:100)
        num_items = rand(1:10000)
        rand_x = rand(bins_x, num_items)
        rand_y = rand(bins_y, num_items)
        R = DeconvUtil.fit_R(rand_y, rand_x, bins_y=bins_y, bins_x=bins_x)
        x_hist = DeconvUtil.fit_pdf(rand_x, bins_x, normalize=false)
        y_hist = DeconvUtil.fit_pdf(rand_y, bins_y, normalize=false)
        @test x_hist ≈ R * y_hist  atol=1e-6
    end
end


# polynomial_smoothing
@testset "Polynomial smoothing" begin
    for i in 1:10
        
        # simple order 1 check
        num_bins = rand(100:1000)
        f_rand   = DeconvUtil.normalizepdf(rand(num_bins))
        f_smooth = DeconvUtil.polynomial_smoothing(1)(f_rand) # apply smoothing of order 1 to f_rand
        diffs = f_smooth[2:end] - f_smooth[1:end-1]     # array of finite differences
        @test all(isapprox.(diffs, mean(diffs)))        # all differences approximately equal
        
        # multiple smoothings return approximately same array
        smoothing = DeconvUtil.polynomial_smoothing(i)
        f_smooth  = smoothing(f_rand)
        @test isapprox(f_smooth, smoothing(f_smooth)) # twice smoothing does not change result
        
    end
end


# expansion / reduction
@testset "Expansion / reduction" begin
    for _ in 1:10
        
        # random discretizer
        num_bins = rand(10:100)
        edges = sort(rand(num_bins + 1)) # random bin edges
        disc  = LinearDiscretizer(edges)
        bins  = 1:length(bincenters(disc)) # bin indices
        @test extrema(disc) == extrema(edges)
        @test length(bins) == num_bins
        
        # expansion
        factor   = rand(1:10)
        disc_exp = DeconvUtil.expansion_discretizer(disc, factor)
        bins_exp = 1:length(bincenters(disc_exp))
        @test length(bins)     == length(edges) - 1
        @test length(bins_exp) == length(bins) * factor
                
        # reduction
        f_rand = DeconvUtil.normalizepdf(rand(length(bins_exp))) # random expanded pdf
        f_red  = DeconvUtil.reduce(f_rand, factor, normalize = false)
        @test length(f_red) == num_bins
        @test f_red == map(i -> sum(f_rand[i:(i + factor - 1)]), 1:factor:length(f_rand))
        
        f_red_full = DeconvUtil.reduce(f_rand, factor, true, normalize = false)
        @test all(map(i -> all(isapprox.(f_red_full[i:(i + factor - 1)],
                                         mean(f_red_full[i:(i + factor - 1)]))),
                      1:factor:length(f_red_full)))
        
    end
end


# chi2s
@testset "Chi Square distance" begin
    for _ in 1:10
        num_bins  = rand(1:100)
        a = rand(-100:1000, num_bins)
        b = rand(0:100, num_bins)
        @test DeconvUtil.chi2s(a, b) >= 0
        
        a = zeros(num_bins)
        b = zeros(num_bins)
        a[1] = 1
        b[1] = 1
        last = DeconvUtil.chi2s(a, b)
        for _ in 1:10
            a[rand(2:num_bins)] += rand(1:10)
            next = DeconvUtil.chi2s(a, b)
            @test next > last
            last = next
        end
    end
end

@testset "Chi Square equality to Distances.jl" begin
    for _ in 1:100
        num_bins  = rand(1:100)
        a = rand(num_bins)
        b = rand(num_bins)
        @test DeconvUtil.chi2s(a, b, false) ≈ 2 * Distances.chisq_dist(a, b) atol=1e-6
    end
end


end # end of util testset