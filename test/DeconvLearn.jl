# 
# Unit tests for the Sklearn module
# 
@testset "CherenkovDeconvolution.DeconvLearn, as tested in test/DeconvLearn.jl" begin


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


@testset "DeconvLearn.TreeDiscretizer" begin
    td = DeconvLearn.TreeDiscretizer(X_train, y_train, 3)
    x_data = DeconvLearn.encode(td, X_data)
    @test x_data[1] == x_data[2]
    @test x_data[3] == x_data[4]
    @test x_data[5] == x_data[6]
    @test DeconvLearn.bins(td) == [ 1, 2, 3 ]
end

@testset "DeconvLearn.KMeansDiscretizer" begin
    kd = DeconvLearn.KMeansDiscretizer(X_train, 3)
    x_data = DeconvLearn.encode(kd, X_data)
    @test x_data[1] == x_data[2]
    @test x_data[3] == x_data[4]
    @test x_data[5] == x_data[6]
    @test DeconvLearn.bins(kd) == [ 1, 2, 3 ]
end


end # end of sklearn testset
