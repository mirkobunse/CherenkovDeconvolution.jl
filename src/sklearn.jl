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
info("ScikitLearn utilities are available in CherenkovDeconvolution")
using YAML, ScikitLearn
using ScikitLearnBase.weighted_sum, PyCall

@sk_import naive_bayes : GaussianNB
@sk_import tree        : DecisionTreeClassifier
@sk_import ensemble    : RandomForestClassifier
@sk_import cluster     : KMeans
@sk_import calibration : CalibratedClassifierCV

export TreeDiscretization


"""
    trainpredict(data, train, configfile, target, weight=nothing)

Train a classifier specified by the `configfile` file using the `train` DataFrame. Predict
the `target` column of the `data` using that classifier.

Return a DataFrame of predictions for the `data` input.
Each returned DataFrame will contain the confidence distribution of each example.
If a `weight` column is given in `train`, it is used as an instance weight vector.
"""
function trainpredict(data::AbstractDataFrame, train::AbstractDataFrame, configfile::String,
                      target::Symbol, weight::Union{Symbol, Void} = nothing; calibrate::Bool = false)
    
    # prepare training set
    features = setdiff(names(train), 
                       weight != nothing ? [target, weight] : [target])
    X_train  = convert(Array{Float64,2}, train[:, features])
    y_train  = map(Symbol, train[target])
    
    # train classifier
    classifier = _from_config(configfile)
    if calibrate
        classifier = CalibratedClassifierCV(classifier, method="isotonic")
    end
    if weight != nothing
        ScikitLearn.fit!(classifier, X_train, y_train; sample_weight = train[weight])
    else
        ScikitLearn.fit!(classifier, X_train, y_train)
    end
    
    # apply to data
    X_data   = convert(Array{Float64,2}, data[:, features])
    mat_prob = ScikitLearn.predict_proba(classifier, X_data) # matrix of probabilities
    
    # matrix to DataFrame
    classes = map(Symbol, get_classes(classifier)) # get_classes gives the correct order
    columns = [ mat_prob[:,j] for j in 1:size(mat_prob,2) ] # array of arrays
    return DataFrame(prediction = parse.(Float64, ScikitLearn.predict(classifier, X_data));
                     zip(classes, columns)...)
    
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
    indexmap::Dict{Int64,Int64}
end

function TreeDiscretization(train::DataFrame, target::Symbol, max_num_leaves::Int,
                            criterion::String="gini"; seed::UInt32 = rand(UInt32))
    
    # prepare training set
    features = setdiff(names(train), [target])
    X_train  = convert(Array{Float64,2}, train[:, features])
    y_train  = map(Symbol, train[target])
    
    # train classifier
    classifier = DecisionTreeClassifier(max_leaf_nodes = max_num_leaves,
                                        criterion = criterion, random_state = seed)
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

# obtain a classifier from a configuration (only imported classes are available)
function _from_config(configfile::String)
    c = YAML.load_file(configfile) # read config
    Classifier = eval(parse(c["classifier"])) # constructor method
    parameters = c["parameters"] != nothing ? c["parameters"] : Dict{Symbol,Any}() # ensure Dict type
    
    # parameters as keyword arguments in constructor
    return Classifier(; zip(map(Symbol, keys(parameters)), values(parameters))...)
end

