# 
# CherenkovDeconvolution.jl
# Copyright 2018, 2019 Mirko Bunse
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
module Sklearn

using DataFrames, ScikitLearn, Discretizers
using PyCall: PyObject, PyArray, pycall, pyimport
import CherenkovDeconvolution.Util

export ClusterDiscretizer, TreeDiscretizer, KMeansDiscretizer
export train_and_predict_proba, encode, bins


# the following replacement for @sk_import enables precompilation
const __KMeans = Ref{PyObject}()
const __DecisionTreeClassifier = Ref{PyObject}()

function __init__()
    global __KMeans[] = pyimport("sklearn.cluster").KMeans
    global __DecisionTreeClassifier[] = pyimport("sklearn.tree").DecisionTreeClassifier
end

KMeans(args...; kwargs...) = __KMeans[](args...; kwargs...)
DecisionTreeClassifier(args...; kwargs...) = __DecisionTreeClassifier[](args...; kwargs...)


"""
    train_and_predict_proba(classifier, :sample_weight)

Obtain a `train_and_predict_proba` object for DSEA.

The optional argument gives the name of the `classifier` parameter with which the sample
weight can be specified when calling `ScikitLearn.fit!`. Usually, its value does not need to
be changed. However, if for example a scikit-learn `Pipeline` object is the `classifier`,
the name of the step has to be provided like `:stepname__sample_weight`.
"""
function train_and_predict_proba(classifier, sample_weight::Union{Symbol,Nothing}=:sample_weight)
    return (X_data::Array, X_train::Array, y_train::Vector, w_train::Vector) -> begin
        kwargs_fit = sample_weight == nothing ? [] : [ (sample_weight, Util.normalizepdf(w_train)) ]
        ScikitLearn.fit!(classifier, X_train, y_train; kwargs_fit...)
        return ScikitLearn.predict_proba(classifier, X_data) # matrix of probabilities
    end
end


"""
    abstract type ClusterDiscretizer <: AbstractDiscretizer

Supertype of any clustering-based discretizer mapping from an n-dimensional space to a
single cluster index dimension.
"""
abstract type ClusterDiscretizer{T<:Number} <: AbstractDiscretizer{Vector{T}, Int} end

@doc """
    bins(d::T) where T <: ClusterDiscretizer

Return the bin indices of `d`.
""" bins # overwritten by sub-types


struct TreeDiscretizer{T<:Number} <: ClusterDiscretizer{T}
    model::PyObject
    indexmap::Dict{Int64,Int64}
end

"""
    TreeDiscretizer(X_train, y_train, J, criterion="gini"; seed)

A decision tree with at most `J` leaves is trained on `X_train` to predict `y_train`. This
tree is used to discretize multidimensional data with `encode()`.
"""
function TreeDiscretizer(X_train::AbstractMatrix{TN},
                         y_train::AbstractVector{TI},
                         J::TI, criterion::String="gini";
                         seed::Integer=rand(UInt32)) where {TN<:Number, TI<:Int}
    # conversion required for ScikitLearn
    X_train_c = convert(Matrix, X_train)
    y_train_c = convert(Vector, y_train)
    
    # train classifier
    classifier = DecisionTreeClassifier(max_leaf_nodes = convert(UInt32, J),
                                        criterion      = criterion,
                                        random_state   = convert(UInt32, seed))
    ScikitLearn.fit!(classifier, X_train_c, y_train_c)
    
    # create some "nice" indices 1,...n
    x_train = _apply(classifier, X_train_c) # leaf indices, which are rather arbitrary
    indexmap = Dict(zip(unique(x_train), 1:length(unique(x_train))))
    return TreeDiscretizer{TN}(classifier, indexmap)
end

"""
    encode(d::TreeDiscretizer, X_data)

Discretize `X_data` using the leaf indices in the decision tree of `d` as discrete values.
"""
function Discretizers.encode(d::TreeDiscretizer{T}, X_data::AbstractMatrix{T}) where T<:Number
    x_data = _apply(d.model, convert(Matrix, X_data))
    return map(i -> d.indexmap[i], convert(Vector{Int64}, x_data))
end

_apply(model::PyObject, X) = pycall(model.apply, PyArray, X) # return the leaf indices of X

bins(d::TreeDiscretizer) = sort(collect(values(d.indexmap)))


struct KMeansDiscretizer{T<:Number} <: ClusterDiscretizer{T}
    model::PyObject
    k::Int64
end

"""
    KMeansDiscretizer(X_train, k)

Unsupervised clustering using all columns in `train`, finding `k` clusters.
It can be used to `discretize()` multidimensional data.
"""
function KMeansDiscretizer(X_train::AbstractMatrix{T}, k::Int;
                           seed::UInt32=rand(UInt32)) where T<:Number
    clustering = KMeans(n_clusters=k, n_init=1, random_state=seed)
    ScikitLearn.fit!(clustering, convert(Matrix, X_train))
    return KMeansDiscretizer{T}(clustering, k)
end

"""
    encode(d::KMeansDiscretizer, X_data)

Discretize `X_data` using the cluster indices of `d` as discrete values.
"""
Discretizers.encode(d::KMeansDiscretizer{T}, X_data::AbstractMatrix{T}) where T<:Number =
    convert(Vector{Int64}, ScikitLearn.predict(d.model, convert(Matrix, X_data))) .+ 1

bins(d::KMeansDiscretizer) = collect(1:d.k)


end

