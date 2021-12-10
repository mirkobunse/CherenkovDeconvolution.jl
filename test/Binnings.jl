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
               [ 0, 1, 0 ])' # LinearAlgebra.Adjoint{Int64, Matrix{Int64}}
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

# sparse copies from Julia and Python
X_train_sparse = sparse(X_train)
X_data_sparse = sparse(X_data)
X_train_scipy = SparseMatrixCSC(SCIPY_SPARSE.csc_matrix(X_train))
X_data_scipy = SparseMatrixCSC(SCIPY_SPARSE.csc_matrix(X_data))

@testset "Binnings.TreeBinning" begin
    seed = rand(UInt32)

    Random.seed!(seed)
    td = BinningDiscretizer(TreeBinning(3), X_train, y_train)
    x_data = encode(td, X_data)
    @test x_data[1] == x_data[2]
    @test x_data[3] == x_data[4]
    @test x_data[5] == x_data[6]
    @test bins(td) == [ 1, 2, 3 ]

    Random.seed!(seed)
    td_sparse = BinningDiscretizer(TreeBinning(3), X_train_sparse, y_train)
    x_data_sparse = encode(td, X_data_sparse)
    @test x_data == x_data_sparse

    Random.seed!(seed)
    td_scipy = BinningDiscretizer(TreeBinning(3), X_train_scipy, y_train)
    x_data_scipy = encode(td, X_data_scipy)
    @test x_data == x_data_scipy
end

@testset "Binnings.KMeansBinning" begin
    seed = rand(UInt32)

    Random.seed!(seed)
    kd = BinningDiscretizer(KMeansBinning(3), X_train, y_train)
    x_data = encode(kd, X_data)
    @test x_data[1] == x_data[2]
    @test x_data[3] == x_data[4]
    @test x_data[5] == x_data[6]
    @test bins(kd) == [ 1, 2, 3 ]

    Random.seed!(seed)
    kd_sparse = BinningDiscretizer(KMeansBinning(3), X_train_sparse, y_train)
    x_data_sparse = encode(kd, X_data_sparse)
    @test x_data == x_data_sparse

    Random.seed!(seed)
    kd_scipy = BinningDiscretizer(KMeansBinning(3), SparseMatrixCSC(X_train_scipy), y_train)
    x_data_scipy = encode(kd, X_data_scipy)
    @test x_data == x_data_scipy
end


end # testset
