# 
# Unit tests for the Binnings module
# 
@testset "CherenkovDeconvolution.Binnings, as tested in test/Binnings.jl" begin

# dummy data
X_train = hcat([ 0, 0, 1 ],
               [ 0, 0, 1 ],
               [ 1, 1, 1 ],
               [ 1, 1, 1 ],
               [ 0, 1, 0 ],
               [ 0, 1, 0 ])'
y_train = [ 1, 1, 2, 2, 3, 3 ]
X_data = hcat([ 1, 1, 0 ],
              [ 1, 1, 0 ],
              [ 1, 0, 1 ],
              [ 1, 0, 1 ],
              [ 0, 0, 0 ],
              [ 0, 0, 0 ])'

# shuffle
train_order = randperm(size(X_train, 1))
X_train = X_train[train_order, :]
y_train = y_train[train_order]

@testset "Binnings.TreeBinning" begin
    td = BinningDiscretizer(TreeBinning(3), X_train, y_train)
    x_data = encode(td, X_data)
    @test x_data[1] == x_data[2]
    @test x_data[3] == x_data[4]
    @test x_data[5] == x_data[6]
    @test bins(td) == [ 1, 2, 3 ]
end

@testset "Binnings.KMeansBinning" begin
    kd = BinningDiscretizer(KMeansBinning(3), X_train, y_train)
    x_data = encode(kd, X_data)
    @test x_data[1] == x_data[2]
    @test x_data[3] == x_data[4]
    @test x_data[5] == x_data[6]
    @test bins(kd) == [ 1, 2, 3 ]
end


end # testset
