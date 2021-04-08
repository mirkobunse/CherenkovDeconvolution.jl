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

using DataFrames, ScikitLearn, Discretizers
using PyCall: PyObject, PyArray, pycall, pyimport

export Binning, BinningDiscretizer, bins, encode, KMeansBinning, TreeBinning

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
    TreeBinning(J; kwargs...)

A supervised tree binning strategy with up to `J` clusters.

### Keyword arguments

- `criterion = "gini"` is the splitting criterion of the tree.
- `seed = rand(UInt32)` is the random seed for tie breaking.
"""
struct TreeBinning <: Binning
    J :: Int
    criterion :: String
    seed :: Int
    TreeBinning(
            J::Integer;
            criterion::AbstractString="gini",
            seed::Integer=rand(UInt32)) =
        new(J, criterion, seed)
end

"""
    KMeansBinning(J; seed=rand(UInt32))

An unsupervised binning strategy with up to `J` clusters.
"""
struct KMeansBinning <: Binning
    J :: Int
    seed :: Int
    KMeansBinning(
            J::Integer;
            seed::Integer=rand(UInt32)) =
        new(J, seed)
end

"""
    abstract type BinningDiscretizer <: AbstractDiscretizer

Supertype of any clustering-based discretizer mapping from an n-dimensional space to a
single cluster index dimension.
"""
abstract type BinningDiscretizer{T<:Any} <: AbstractDiscretizer{Vector{T}, Int} end

"""
    struct TreeDiscretizer <: BinningDiscretizer

A discretizer that is trained with a `TreeBinning` strategy.
"""
struct TreeDiscretizer{T<:Number} <: BinningDiscretizer{T}
    model::PyObject
    indexmap::Dict{Int64,Int64}
end

# constructor of a TreeDiscretizer
function BinningDiscretizer(
        b::TreeBinning,
        X_trn::AbstractMatrix{T},
        y_trn::AbstractVector{I}
        ) where {T<:Number, I<:Integer}
    X_trn_c = convert(Matrix, X_trn) # sklearn requires conversion
    classifier = DecisionTreeClassifier(
        max_leaf_nodes = convert(UInt32, b.J),
        criterion = b.criterion,
        random_state = convert(UInt32, b.seed)
    )
    ScikitLearn.fit!(classifier, X_trn_c, convert(Vector, y_trn))
    
    # create some "nice" indices 1,...n
    x_trn = _apply(classifier, X_trn_c) # leaf indices, which are rather arbitrary
    indexmap = Dict(zip(unique(x_trn), 1:length(unique(x_trn))))
    return TreeDiscretizer{T}(classifier, indexmap)
end

"""
    encode(d::TreeDiscretizer, X_obs)

Discretize `X_obs` using the leaf indices in the decision tree of `d` as discrete values.
"""
function Discretizers.encode(d::TreeDiscretizer{T}, X_obs::AbstractMatrix{T}) where T<:Number
    x_data = _apply(d.model, convert(Matrix, X_obs))
    return map(i -> d.indexmap[i], convert(Vector{Int64}, x_data))
end

_apply(model::PyObject, X) = pycall(model.apply, PyArray, X) # return the leaf indices of X

bins(d::TreeDiscretizer) = sort(collect(values(d.indexmap)))

"""
    struct KMeansDiscretizer <: BinningDiscretizer

A discretizer that is trained with a `KMeansBinning` strategy.
"""
struct KMeansDiscretizer{T<:Number} <: BinningDiscretizer{T}
    model::PyObject
    J::Int
end

function BinningDiscretizer(
        b::KMeansBinning,
        X_trn::AbstractMatrix{T},
        y_trn::AbstractVector{I} = Int[] # y_trn is optional, here
        ) where {T<:Number, I<:Integer}
    clustering = KMeans(n_clusters=b.J, n_init=1, random_state=b.seed)
    ScikitLearn.fit!(clustering, convert(Matrix, X_trn))
    return KMeansDiscretizer{T}(clustering, b.J)
end

"""
    encode(d::KMeansDiscretizer, X_obs)

Discretize `X_obs` using the cluster indices of `d` as discrete values.
"""
Discretizers.encode(d::KMeansDiscretizer{T}, X_obs::AbstractMatrix{T}) where T<:Number =
    convert(Vector{Int64}, ScikitLearn.predict(d.model, convert(Matrix, X_obs))) .+ 1

bins(d::KMeansDiscretizer) = collect(1:d.J)

@doc """
    bins(d::T) where T <: BinningDiscretizer

Return the bin indices of `d`.
""" bins # update the documentation

end # module
