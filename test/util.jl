# 
# Unit tests for the Util module
# 


# dummy data
y = [2, 1, 3, 3, 3, 1]
Rdict = Dict(1 => 1,
             2 => 1,
             3 => 2,
             4 => 2)
x = get.(Rdict, y, -1)
R = zeros(Float64, (2, 3))
for i in 1:length(y)
    R[x[i], y[i]] += 1
end
R = Util.normalizetransfer(R)


# normalizepdf
@testset "Normalize pdfs" begin
    for _ in 1:10
        num_bins  = rand(1:100)
        num_items = rand(1:1000)
        @test sum(Util.normalizepdf(num_items .* rand(num_bins))) ≈ 1  atol=1e-6
    end
end


# pdfs
@test Util.fit_pdf(y) == [ 1/3, 1/6, 1/2 ]

@testset "Random pdfs" begin
    for _ in 1:10
        
        # random array
        num_bins  = rand(1:100)
        num_items = rand(1:1000)
        rand_arr = rand(1:num_bins, num_items)
        
        # normalized pdf
        rand_pdf = Util.fit_pdf(rand_arr, 1:num_bins)
        @test sum(rand_pdf) ≈ 1  atol=1e-6
        @test length(rand_pdf) == num_bins
        
        # unnormalized (histogram)
        rand_hist = Util.fit_pdf(rand_arr, 1:num_bins, normalize = false)
        @test sum(rand_hist) == num_items
        @test length(rand_hist) == num_bins
        
        # laplace correction
        rand_arr[rand_arr .== 1] = 2 # replace full class, which consequently gets probability zero
        @test Util.fit_pdf(rand_arr, 1:num_bins, normalize = false)[1] == 0
        @test Util.fit_pdf(rand_arr, 1:num_bins, laplace = true, normalize = false)[1] == 1
    end
end


# fit_R
@test size(Util.fit_R(y, x)) == (2, 3)
@test Util.fit_R(y, x) == R

@testset "Random transfer matrices" begin
    for _ in 1:10
        bins_y = 1:rand(1:100)
        bins_x = 1:rand(1:100)
        num_items = rand(1:10000)
        rand_x = rand(bins_x, num_items)
        rand_y = rand(bins_y, num_items)
        R = Util.fit_R(rand_y, rand_x, bins_y=bins_y, bins_x=bins_x)
        x_hist = Util.fit_pdf(rand_x, bins_x, normalize=false)
        y_hist = Util.fit_pdf(rand_y, bins_y, normalize=false)
        @test x_hist == R * y_hist
    end
end


# TODO smoothpdf


# chi2s
@testset "Chi Square distance" begin
    for _ in 1:10
        num_bins  = rand(1:100)
        a = rand(-100:1000, num_bins)
        b = rand(0:100, num_bins)
        @test Util.chi2s(a, b) >= 0
        
        a = zeros(num_bins)
        b = zeros(num_bins)
        a[1] = 1
        b[1] = 1
        last = Util.chi2s(a, b)
        for _ in 1:10
            a[rand(2:num_bins)] += rand(1:10)
            next = Util.chi2s(a, b)
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
        @test Util.chi2s(a, b, false) ≈ 2 * Distances.chisq_dist(a, b) atol=1e-6
    end
end

