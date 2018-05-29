# 
# Unit tests for the Util module
# 


# dummy data
y = [2, 1, 4, 4, 4, 1]
Rdict = Dict(1 => 1,
             2 => 1,
             3 => 3,
             4 => 3)
x = get.(Rdict, y, -1)
R = zeros(Float64, (3, 4))
for i in 1:length(y)
    R[x[i], y[i]] += 1
end
Util.normalizetransfer!(R)


# histograms
@test Util.histogram(y)      == [2, 1, 3]
@test Util.histogram(y, 1:4) == [2, 1, 0, 3]

@testset "Random histograms" for _ in 1:10
    num_bins  = rand(1:100)
    num_items = rand(1:1000)
    rand_hist = Util.histogram(rand(1:num_bins, num_items), 1:num_bins)
    @test sum(rand_hist) == num_items
    @test length(rand_hist) == num_bins
end


# normalizepdf
@testset "Normalize pdfs" for _ in 1:10
    num_bins  = rand(1:100)
    num_items = rand(1:1000)
    @test sum(Util.normalizepdf(num_items .* rand(num_bins))) ≈ 1  atol=1e-6
end


# empiricaltransfer
@test size(Util.empiricaltransfer(y, x)) == (2, 3)
@test Util.empiricaltransfer(y, x, ylevels = 1:4, xlevels = 1:3) == R


# TODO smoothpdf


# chi2s
@testset "Chi Square distance" for _ in 1:10
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

@testset "Chi Square equality to Distances.jl" for _ in 1:100
    num_bins  = rand(1:100)
    a = rand(num_bins)
    b = rand(num_bins)
    @test Util.chi2s(a, b, false) ≈ 2 * Distances.chisq_dist(a, b) atol=1e-6
end

