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
using DataFrames, Requires, ScikitLearn
using ScikitLearnBase.weighted_sum, PyCall
import CherenkovDeconvolution.Util

@sk_import naive_bayes : GaussianNB
@sk_import tree        : DecisionTreeClassifier
@sk_import ensemble    : RandomForestClassifier
@sk_import cluster     : KMeans
@sk_import calibration : CalibratedClassifierCV

export TreeDiscretization


"""
    train_and_predict_proba(classname, calibrate = false; kwargs...)

Obtain a `train_and_predict_proba` object for DSEA. The following `classname`s are
available:

- GaussianNB
- DecisionTreeClassifier
- RandomForestClassifier

The keyword arguments configure these classifiers (see official scikit-learn doc).
"""
function train_and_predict_proba(classifier)
    return (X_data, X_train, y_train, w_train, ylevels) -> begin
        ScikitLearn.fit!(classifier, X_train, y_train; sample_weight = w_train)
        proba = ScikitLearn.predict_proba(classifier, X_data) # matrix of probabilities
        
        # permute columns in order of ylevels
        classes = map(string, get_classes(classifier)) # get_classes gives the actual order
        return proba[:, map(i -> findfirst(classes .== string(ylevels[i])), 1:length(ylevels)) ]
    end
end

"""
    classifier(classname, calibrate = false; kwargs...)

Obtain a classifier to be used in `train_and_predict_proba`. The following values of
`classname` are available:

- GaussianNB
- DecisionTreeClassifier
- RandomForestClassifier

The keyword arguments configure the corresponding class (see official scikit-learn doc).
"""
function classifier(classname::AbstractString,
                    calibrate::Bool = false;
                    kwargs...)
    Classifier = eval(parse(classname)) # constructor method
    classifier = Classifier(; kwargs...)
    if calibrate
        classifier = CalibratedClassifierCV(classifier, method="isotonic")
    end
    return classifier
end

# train_and_predict_proba result from configuration file
@require YAML begin
    import YAML
    """
        from_config(configfile)
    
    Obtain the result of the `classifier` function from a YAML configuration instead of
    using function arguments.
    """
    function classifier_from_config(configfile::AbstractString)
        c = YAML.load_file(configfile) # read config
        classname = c["classifier"]
        params    = get(c, "parameters", nothing) != nothing ? c["parameters"] : Dict{Symbol, Any}()
        calibrate = get(c, "calibrate",  false)
        return classifier(classname, calibrate;
                          zip(map(Symbol, keys(params)), values(params))...)
    end
end


"""
    trainpredict(data, train, configfile, y, w=nothing)

Train a classifier specified by the `configfile` file using the `train` DataFrame. Predict
the `y` column of the `data` using that classifier.

Return a DataFrame of predictions for the `data` input.
Each returned DataFrame will contain the confidence distribution of each example.
If a `w` column is given in `train`, it is used as an instance weight vector.
""" # TODO DEPRECATED
function trainpredict(data::AbstractDataFrame, train::AbstractDataFrame, configfile::String,
                      y::Symbol, w::Union{Symbol, Void} = nothing;
                      ylevels::AbstractArray = sort(unique(train[y])))
    # prepare matrices
    features = setdiff(names(train), w != nothing ? [y, w] : [y])
    X_data,  _       = Util.df2Xy(data,  y, features)
    X_train, y_train = Util.df2Xy(train, y, features)
    w_train  = w != nothing ? train[w] : ones(size(train, 1))
    
    # configure and apply function object
    trainpredict = train_and_predict_proba(classifier_from_config(configfile))
    mat_prob     = trainpredict(X_data, X_train, y_train, w_train, ylevels)
    
    # matrix to DataFrame (zip levels with array of column arrays)
    return Util.prob2df(mat_prob, ylevels)
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
