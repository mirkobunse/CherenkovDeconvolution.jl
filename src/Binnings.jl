# 
# CherenkovDeconvolution.jl
# Copyright 2018-2021 Mirko Bunse
# 
# 
# Deconvolution methods for Cherenkov astronomy and other use cases in experimental physics.
# 
# 
# CherenkovDeconvolution.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CherenkovDeconvolution.jl.  If not, see <http://www.gnu.org/licenses/>.
# 
module Binnings

using ScikitLearn, Discretizers
using PyCall: PyObject, PyArray, pycall, pyimport

export
    Binning,
    BinningDiscretizer,
    BinningPreprocessor,
    bins,
    ClassificationPreprocessor,
    DefaultPreprocessor,
    encode,
    KMeansBinning,
    TreeBinning

# the following replacement for @sk_import enables precompilation
const __KMeans = Ref{PyObject}()
const __DecisionTreeClassifier = Ref{PyObject}()

function __init__()
    ScikitLearn.Skcore.import_sklearn() # make sure sklearn is installed
    global __KMeans[] = pyimport("sklearn.cluster").KMeans
    global __DecisionTreeClassifier[] = pyimport("sklearn.tree").DecisionTreeClassifier
end

KMeans(args...; kwargs...) = __KMeans[](args...; kwargs...)
DecisionTreeClassifier(args...; kwargs...) = __DecisionTreeClassifier[](args...; kwargs...)

"""
    abstract type Binning

Supertype of all binning strategies for observable features.
"""
abstract type Binning end

"""
    abstract type BinningPreprocessor

Supertype of all preprocessing techniques that are applied to observable features
before the actual `Binning`.
"""
abstract type BinningPreprocessor end

"""
    fit!(binning_preprocessor, X_trn, y_trn)

Fit the `binning_preprocessor` to the training data `(X_trn, y_trn)`.
"""
fit!(p::BinningPreprocessor, X::Any, y::AbstractVector{I}) where {I<:Integer} = p

"""
    transform(binning_preprocessor, X_obs)

Apply the `binning_preprocessor` to the observed data `X_obs`.
"""
transform(p::BinningPreprocessor, X::Any, y::AbstractVector{I}) where {I<:Integer} =
    throw(ArgumentError("Implementation missing for $(typeof(p))")) # must be implemented for sub-types

"""
    type DefaultPreprocessor <: BinningPreprocessor

A default preprocessor that does not transform the data.
"""
struct DefaultPreprocessor <: BinningPreprocessor end
transform(p::DefaultPreprocessor, X::Any) = X # do nothing

"""
    ClassificationPreprocessor(classifier)

The output of a `classifier` is used as the input of the actual `Binning`.
"""
struct ClassificationPreprocessor <: BinningPreprocessor
    classifier::Any
end
fit!(p::ClassificationPreprocessor, X::Any, y::AbstractVector{I}) where {I<:Integer} =
    ScikitLearn.fit!(p.classifier, X, y)
transform(p::ClassificationPreprocessor, X::Any) =
    ScikitLearn.predict_proba(p.classifier, X)

"""
    TreeBinning(J, [preprocessor]; kwargs...)

A supervised tree binning strategy with an optional `preprocessor` and up to `J`
clusters.

### Keyword arguments

- `criterion = "gini"` is the splitting criterion of the tree.
- `seed = rand(UInt32)` is the random seed for tie breaking.
"""
struct TreeBinning <: Binning
    J :: Int
    preprocessor :: BinningPreprocessor
    criterion :: String
    seed :: Int
    TreeBinning(
            J::Integer,
            preprocessor::BinningPreprocessor=DefaultPreprocessor();
            criterion::AbstractString="gini",
            seed::Integer=rand(UInt32)) =
        new(J, preprocessor, criterion, seed)
end

"""
    KMeansBinning(J, [preprocessor]; seed=rand(UInt32))

An unsupervised binning strategy with an optional `preprocessor` and up to `J`
clusters.
"""
struct KMeansBinning <: Binning
    J :: Int
    preprocessor :: BinningPreprocessor
    seed :: Int
    KMeansBinning(
            J::Integer,
            preprocessor::BinningPreprocessor=DefaultPreprocessor();
            seed::Integer=rand(UInt32)) =
        new(J, preprocessor, seed)
end

"""
    abstract type BinningDiscretizer

Supertype of any clustering-based discretizer mapping from an n-dimensional space to a
single cluster index dimension.
"""
abstract type BinningDiscretizer end

"""
    struct TreeDiscretizer <: BinningDiscretizer

A discretizer that is trained with a `TreeBinning` strategy.
"""
struct TreeDiscretizer <: BinningDiscretizer
    model::PyObject
    indexmap::Dict{Int,Int}
    preprocessor::BinningPreprocessor
end

# constructor of a TreeDiscretizer
function BinningDiscretizer(
        b::TreeBinning,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer}
    fit!(b.preprocessor, X_trn, y_trn)
    X_trn_prep = transform(b.preprocessor, X_trn)
    classifier = DecisionTreeClassifier(
        max_leaf_nodes = convert(UInt32, b.J),
        criterion = b.criterion,
        random_state = convert(UInt32, b.seed)
    )
    ScikitLearn.fit!(classifier, X_trn_prep, y_trn)

    # create some "nice" indices 1,...n
    x_trn = _apply(classifier, X_trn_prep) # leaf indices, which are rather arbitrary
    indexmap = Dict(zip(unique(x_trn), 1:length(unique(x_trn))))
    return TreeDiscretizer(classifier, indexmap, b.preprocessor)
end

"""
    encode(d::TreeDiscretizer, X_obs)

Discretize `X_obs` using the leaf indices in the decision tree of `d` as discrete values.
"""
function Discretizers.encode(d::TreeDiscretizer, X_obs::Any)
    x_data = _apply(d.model, transform(d.preprocessor, X_obs))
    return map(i -> d.indexmap[i], convert(Vector{Int}, x_data))
end

_apply(model::PyObject, X::Any) = pycall(model.apply, PyArray, X) # return the leaf indices of X

bins(d::TreeDiscretizer) = sort(collect(values(d.indexmap)))

"""
    struct KMeansDiscretizer <: BinningDiscretizer

A discretizer that is trained with a `KMeansBinning` strategy.
"""
struct KMeansDiscretizer <: BinningDiscretizer
    model::PyObject
    J::Int
    preprocessor::BinningPreprocessor
end

function BinningDiscretizer(
        b::KMeansBinning,
        X_trn::Any,
        y_trn::AbstractVector{I} = Int[] # y_trn is optional, here
        ) where {I<:Integer}
    fit!(b.preprocessor, X_trn, y_trn)
    X_trn_prep = transform(b.preprocessor, X_trn)
    clustering = KMeans(n_clusters=b.J, n_init=1, random_state=b.seed)
    ScikitLearn.fit!(clustering, X_trn_prep)
    return KMeansDiscretizer(clustering, b.J, b.preprocessor)
end

"""
    encode(d::KMeansDiscretizer, X_obs)

Discretize `X_obs` using the cluster indices of `d` as discrete values.
"""
Discretizers.encode(d::KMeansDiscretizer, X_obs::Any) =
    convert(
        Vector{Int},
        ScikitLearn.predict(d.model, transform(d.preprocessor, X_obs))
    ) .+ 1

bins(d::KMeansDiscretizer) = collect(1:d.J)

@doc """
    bins(d::T) where T <: BinningDiscretizer

Return the bin indices of `d`.
""" bins # update the documentation

# deprecated syntax
export TreeDiscretizer, KMeansDiscretizer
function TreeDiscretizer(
        X_trn :: AbstractMatrix{TN},
        y_trn :: AbstractVector{TI},
        J     :: TI,
        criterion :: String="gini";
        seed  :: Integer=rand(UInt32)
        ) where {TN<:Number, TI<:Int}
    Base.depwarn(join([
        "`TreeDiscretizer(data, config)` is deprecated; ",
        "call `BinningDiscretizer(TreeBinning(config), data)` instead"
    ]), :TreeDiscretizer)
    binning = TreeBinning(J; criterion=criterion, seed=seed)
    return BinningDiscretizer(binning, X_trn, y_trn)
end
function KMeansDiscretizer(
        X_trn :: AbstractMatrix{TN},
        J     :: TI,
        seed  :: Integer=rand(UInt32)
        ) where {TN<:Number, TI<:Int}
    Base.depwarn(join([
        "`KMeansDiscretizer(data, config)` is deprecated; ",
        "call `BinningDiscretizer(KMeansBinning(config), data)` instead"
    ]), :KMeansDiscretizer)
    binning = KMeansBinning(J; seed=seed)
    return BinningDiscretizer(binning, X_trn)
end

end # module
