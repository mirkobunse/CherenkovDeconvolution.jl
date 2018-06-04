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
module Sklearn

info("ScikitLearn utilities are available in CherenkovDeconvolution")

using DataFrames, Requires, ScikitLearn, PyCall.PyObject, ScikitLearnBase.weighted_sum
import CherenkovDeconvolution.Util

@sk_import tree        : DecisionTreeClassifier
@sk_import cluster     : KMeans


export ClusterDiscretization, TreeDiscretization, KMeansDiscretization


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


abstract type ClusterDiscretization end

"""
    TreeDiscretization(train, target, max_num_leaves, criterion = "gini")

A decision tree is trained on the `train` set to predict the `target` variable and it has
at most `max_num_leaves`. It can be used to `discretize()` multidimensional data.
"""
type TreeDiscretization <: ClusterDiscretization
    model::PyObject
    features::Array{Symbol, 1}
    indexmap::Dict{Int64, Int64}
end

function TreeDiscretization(train::DataFrame, target::Symbol, max_num_leaves::Int,
                            criterion::String="gini"; seed::UInt32 = rand(UInt32))
    
    # prepare training set
    features = setdiff(names(train), [target])
    X_train  = convert(Array{Float64, 2}, train[:, features])
    y_train  = map(Symbol, train[target])
    
    # train classifier
    classifier = DecisionTreeClassifier(max_leaf_nodes = max_num_leaves,
                                        criterion      = criterion,
                                        random_state   = seed)
    ScikitLearn.fit!(classifier, X_train, y_train)
    
    # create some "nice" indices 1,...n
    X_leaves = _apply(classifier, X_train) # leaf indices, which are rather arbitrary
    indexmap = Dict(zip(unique(X_leaves), 1:length(unique(X_leaves))))
    return TreeDiscretization(classifier, features, indexmap)
    
end

"""
    discretize(data, discr)

Discretize the `data` by using its leaf indices in the decision tree of `discr` as discrete
values. `discr` is obtained from `TreeDiscretization()`.
"""
function discretize(data::DataFrame, discr::TreeDiscretization)
    X_data   = convert(Array{Float64,2}, data[:, discr.features])
    X_leaves = _apply(discr.model, X_data)
    return map(i -> discr.indexmap[i], convert(Array{Int64, 1}, X_leaves))
end

_apply(model::PyObject, X) = pycall(model[:apply], PyArray, X) # return the leaf indices of X

"""
    levels(discr)

Return the discrete levels of `discr`.
"""
levels(discr::TreeDiscretization) = sort(collect(values(discr.indexmap)))

"""
    KMeansDiscretization(train, k)

Unsupervised clustering using all columns in `train`, finding `k` clusters.
It can be used to `discretize()` multidimensional data.
"""
type KMeansDiscretization <: ClusterDiscretization
    model::PyObject
    k::Int64
end

function KMeansDiscretization(train::DataFrame, k::Int)
    X_train = convert(Array{Float64,2}, train)
    clustering = KMeans(n_clusters = k, n_init = 1, random_state = rand(UInt32))
    ScikitLearn.fit!(clustering, X_train)
    return KMeansDiscretization(clustering, k)
end

"""
    discretize(data, discr)

Discretize the `data` by using its cluster indices in the clustering `discr` as discrete
values. `discr` is obtained from `KMeansDiscretization()`.
"""
discretize(data::DataFrame, discr::KMeansDiscretization) =
    convert(Array{Int64, 1}, ScikitLearn.predict(discr.model, convert(Array{Float64,2}, data))) .+ 1

levels(discr::KMeansDiscretization) = collect(1:discr.k)


end
