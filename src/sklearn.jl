# 
# CherenkovDeconvolution.jl
# Copyright 2018 Mirko Bunse
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
using DataFrames, Requires, ScikitLearn, Discretizers
using PyCall: PyObject, PyArray, pycall
using ScikitLearnBase: weighted_sum
import CherenkovDeconvolution.Util

@sk_import tree        : DecisionTreeClassifier
@sk_import cluster     : KMeans


export ClusterDiscretizer, TreeDiscretizer, KMeansDiscretizer
export train_and_predict_proba, encode, bins


"""
    train_and_predict_proba(classifier)

Obtain a `train_and_predict_proba` object for DSEA.
"""
function train_and_predict_proba(classifier::PyObject)
    return (X_data::Matrix, X_train::Matrix, y_train::Array, w_train::Array) -> begin
        ScikitLearn.fit!(classifier, X_train, y_train; sample_weight = w_train)
        return ScikitLearn.predict_proba(classifier, X_data) # matrix of probabilities
    end
end


abstract type ClusterDiscretizer{T<:Number} <: AbstractDiscretizer{Array{T,1},Int} end

"""
    bins(d::ClusterDiscretizer)

Return the bin indices of `d`.
"""
bins(d::ClusterDiscretizer) = nothing # overwritten by sub-types


"""
    TreeDiscretizer(X_train, y_train, J, criterion="gini"; seed)

A decision tree with at most `J` leaves is trained on `X_train` to predict `y_train`. This
tree is used to discretize multidimensional data with `encode()`.
"""
type TreeDiscretizer{T<:Number} <: ClusterDiscretizer{T}
    model::PyObject
    indexmap::Dict{Int64,Int64}
end

TreeDiscretizer(X_train::AbstractMatrix, y_train::AbstractArray, args...; kwargs...) =
    TreeDiscretizer(convert(Array, X_train), convert(Array, y_train), args...; kwargs...)

function TreeDiscretizer{TN<:Number,TI<:Int}(X_train::Matrix{TN},
                                             y_train::Array{TI,1},
                                             J::TI, criterion::String="gini";
                                             seed::Integer=rand(UInt32))
    # train classifier
    classifier = DecisionTreeClassifier(max_leaf_nodes = J,
                                        criterion      = criterion,
                                        random_state   = convert(UInt32, seed))
    ScikitLearn.fit!(classifier, X_train, y_train)
    
    # create some "nice" indices 1,...n
    x_train = _apply(classifier, X_train) # leaf indices, which are rather arbitrary
    indexmap = Dict(zip(unique(x_train), 1:length(unique(x_train))))
    return TreeDiscretizer{TN}(classifier, indexmap)
end

"""
    encode(d, data)

Discretize the `data` by using the leaf indices in the decision tree of `d` as discrete
values. `d` is obtained from `TreeDiscretizer()`.
"""
function Discretizers.encode{T<:Number}(d::TreeDiscretizer{T}, X_data::AbstractMatrix{T})
    x_data = _apply(d.model, convert(Array, X_data))
    return map(i -> d.indexmap[i], convert(Array{Int64,1}, x_data))
end

_apply(model::PyObject, X) = pycall(model[:apply], PyArray, X) # return the leaf indices of X

bins(d::TreeDiscretizer) = sort(collect(values(d.indexmap)))


"""
    KMeansDiscretizer(X_train, k)

Unsupervised clustering using all columns in `train`, finding `k` clusters.
It can be used to `discretize()` multidimensional data.
"""
type KMeansDiscretizer{T<:Number} <: ClusterDiscretizer{T}
    model::PyObject
    k::Int64
end

function KMeansDiscretizer{T<:Number}(X_train::Matrix{T}, k::Int; seed::UInt32=rand(UInt32))
    clustering = KMeans(n_clusters=k, n_init=1, random_state=seed)
    ScikitLearn.fit!(clustering, X_train)
    return KMeansDiscretizer{T}(clustering, k)
end

"""
    encode(d, data)

Discretize the `data` by using its cluster indices in the clustering `d` as discrete
values. `d` is obtained from `KMeansDiscretizer()`.
"""
Discretizers.encode{T<:Number}(d::KMeansDiscretizer{T}, X_data::Matrix{T}) =
    convert(Array{Int64, 1}, ScikitLearn.predict(d.model, X_data)) .+ 1

bins(d::KMeansDiscretizer) = collect(1:d.k)

