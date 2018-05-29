"""
    dsea(data, train, target; kwargs...)

Unfold the `target` column in the DataFrame `data`, as learned from the DataFrame `train`.

The `target` column has to be discrete, i.e., it has to have a limited number of unique values that
are used as labels for the classifier. All other columns are leveraged in training and prediction.

You can monitor the algorithm's progress by providing the keyword argument `inspect`.
If provided, `inspect` is a function called in every iteration.

# Keyword arguments
- `maxiter::Int = 1` gives the maximum number of DSEA iterations within reweighting.
- `epsilon::Float64 = 0` is the minimum symmetric Chi Square distance between iterations.
- `skconfig::String = 'conf/weka/nb.yml'` points to the configuration file that specifies
  the classification algorithm.
- `prior::Union{AbstractArray{Float64, 1}, Symbol} = :uniform` is the prior spectrum used to
  weight training examples in the first iteration. Possible symbol values are `:uniform` and
  `:trainingset`. Any other prior can be specified as an array.
- `alpha::Union{Float64, Function} = 1.0` is the step size taken in every iteration.
  Can be either a constant value or a function with the signature
  `(k::Int, pk::AbstractArray{Float64,1}, lastspec::AbstractArray{Float64,1} -> Float`,
  where `lastspec` is the estimate of the previous iteration and `pk` is the direction that
  DSEA takes in the current iteration `k`.
- `smoothing::Symbol = :none` can also be set to `:polynomial` to apply polynomial smoothing
  [dagostini2010improved], or any other `method` accepted by `smoothpdf`.
  The operation is neither applied to the initial prior, nor to the final result,
  and not to any of the spectra fed into `inspect`.
- `fixweighting::Bool = false` fixes the reweighting of training examples when the training
  set is not uniformly distributed over the target bins.
- `returncontributions::Bool = false` makes the function return a tuple of estimated spectrum
  and the contribution of each individual item. By default, return the spectrum only.
- `inspect::Function = nothing` is a function `(k::Int, alpha::Float64, chi2s::Float64,
  spectrum::Array) -> Any` called in every iteration.

Any other keyword argument is forwarded to the smoothing operation (if smoothing is applied).
"""
function dsea(data::AbstractDataFrame, train::AbstractDataFrame, y::Symbol;
              features::AbstractArray{Symbol, 1} = setdiff(names(train), [y]),
              kwargs...)
    X_data,  _       = Util.df2Xy(data,  y, features)
    X_train, y_train = Util.df2Xy(train, y, features)
    dsea(X_data, X_train, y_train; kwargs...)
end

function dsea{T <: Number}(
              X_data::AbstractArray{Float64, 2},
              X_train::AbstractArray{Float64, 2},
              y_train::AbstractArray{T, 1};
              maxiter::Int64 = 1,
              epsilon::Float64 = 0.0,
              skconfig::String = "conf/sklearn/nb.yml",
              ylevels::AbstractArray{Float64, 1} = sort(unique(y_train)),
              prior::Union{AbstractArray{Float64, 1}, Symbol} = :uniform,
              alpha::Union{Float64, Function} = 1.0,
              smoothing::Symbol = :none,
              inspect::Union{Function, Void} = nothing,
              fixweighting::Bool = false,
              returncontributions::Bool = false,
              calibrate::Bool = false,
              kwargs...)
    
    m = length(ylevels) # number of classes
    n = length(y_train) # number of training examples
    
    # TODO reintroduce checks
    
    # initial estimate is uniform prior
    f       = ones(m) ./ m # TODO optional argument
    f_train = Util.histogram(y_train, ylevels) ./ m # training set distribution (fixweighting)
    if inspect != nothing
        inspect(0, NaN, NaN, f)
    end
    
    # weight training examples according to prior
    binweights = Util.normalizepdf(fixweighting ? f ./ f_train : f)
    w_train = max.([ binweights[findfirst(ylevels .== t)] for t in y_train ], 1/n)
    
    # unfold
    for k in 1:maxiter
        f_prev = f
        
        # predict data and reconstruct spectrum
        trainpredict = Sklearn.train_and_predict_proba(Sklearn.classifier_from_config(skconfig)) # TODO as argument
        proba        = trainpredict(X_data, X_train, y_train, w_train, ylevels)
        f            = _dsea_reconstruct(proba)
        
        # find and apply step size
        pk     = f - f_prev # direction
        alphak = typeof(alpha) == Float64 ? alpha : alpha(k, pk, f_prev)
        f      = f_prev + alphak * pk
        
        chi2s = Util.chi2s(f_prev, f) # Chi Square distance between iterations
        
        # info("DSEA iteration $k/$maxiter ",
        #      fixweighting || smoothing != :none ? "(" : "",
        #      fixweighting ? "fixed weighting" : "",
        #      fixweighting && smoothing != :none ? ", " : "",
        #      smoothing != :none ? string(smoothing) * " smoothing" : "",
        #      fixweighting || smoothing != :none ? ") " : "",
        #      "uses alpha = $alphak (chi2s = $chi2s)")
        
        # optionally monitor progress
        if inspect != nothing
            inspect(k, alphak, chi2s, f)
        end
        
        # stop when convergence is assumed
        if chi2s < epsilon # also holds when alpha is zero
            # info("DSEA convergence assumed from chi2s = $chi2s < epsilon = $epsilon")
            break
        end
        
        # reweighting of items
        if k < maxiter # only done if there is a next iteration
            f = Util.smoothpdf(f, smoothing; kwargs...) # smoothing is an intermediate step
            binweights = Util.normalizepdf(fixweighting ? f ./ f_train : f)
            w_train = max.([ binweights[findfirst(ylevels .== t)] for t in y_train ], 1/n)
        end
        
    end
    
    if returncontributions
        return f, proba
    else
        return f
    end
    
end

# spectral reconstruction: sum of confidences in each bin
_dsea_reconstruct(proba::AbstractArray{Float64, 2}) = map(i -> sum(proba[:, i]), 1:size(proba, 2))

