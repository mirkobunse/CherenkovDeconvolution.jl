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
  spectrum::DataFrame) -> Any` called in every iteration.

Any other keyword argument is forwarded to the smoothing operation (if smoothing is applied).
"""
function dsea(data::AbstractDataFrame, train::AbstractDataFrame, target::Symbol;
              maxiter::Int64 = 1,
              epsilon::Float64 = 0.0,
              skconfig::String = "conf/sklearn/nb.yml",
              ylevels::AbstractArray{Float64, 1} = sort(unique(train[target])),
              prior::Union{AbstractArray{Float64, 1}, Symbol} = :uniform,
              alpha::Union{Float64, Function} = 1.0,
              smoothing::Symbol = :none,
              inspect::Union{Function, Void} = nothing,
              fixweighting::Bool = false,
              returncontributions::Bool = false,
              calibrate::Bool = false,
              kwargs...)
    
    m = length(ylevels) # number of classes
    n = size(train, 1)  # number of training examples
    
    if typeof(prior) <: Symbol && !in(prior, [:uniform, :trainingset])
        error("Illegal value of parameter `prior`: $prior")
    elseif !in(smoothing, [:none, :polynomial])
        error("Illegal value of parameter `smoothing`: $smoothing")
    end
    
    # check that training data and unfolding data fit to each other
    if (length(setdiff(names(train), vcat(names(data), target))) > 0)
        error("""The following attributes are in the training DataFrame, but not in the data DataFrame:
                 $(setdiff(names(train), names(data)))""")
    elseif typeof(prior) <: AbstractArray && length(prior) != m
        error("The given prior is not of the required length $m")
    elseif (m > .05 * size(data, 1))
        warn("""There are $m unique values for $target in the training DataFrame (> 5\% of the data).
                Are you sure the DataFrames are discrete?""")
    end
    
    # initial spectrum
    spec = histogram(train, target; levels = ylevels) # trainingset prior
    wfactor = spec[target] ./ n # weight factor based on training set spectrum (for weighting fix)
    if typeof(prior) <: Symbol && prior == :uniform
        spec[target] = repmat([size(data, 1) / size(spec, 1)], size(spec, 1))
    elseif typeof(prior) <: AbstractArray
        spec[target] = normalizepdf(prior) .* size(data, 1)
    end
    spec[target] = convert(Array{Float64,1}, spec[target]) # ensure type safety
    if (inspect != nothing)
        inspect(0, NaN, NaN, spec)
    end
    
    # weight training examples according to prior
    binweights = normalizepdf(  fixweighting ? spec[target] ./ wfactor : spec[target]  )
    traindf = hcat(DataFrame(train[:, :]),
                   DataFrame(w = max.([ binweights[findfirst(spec[:level] .== t)] for t in train[target] ], 1/size(train, 1))))
    w = names(traindf)[end] # name of weight column (hcat produced view with weights)
    
    # unfold
    preddata = DataFrame()
    for k in 1:maxiter
        lastspec = spec[target]
        
        # predict data and reconstruct spectrum
        preddata = trainpredict(data, traindf, skconfig, target, w, calibrate = calibrate) # from sklearn.jl
        spec[target] = _dsea_reconstruct(preddata, ylevels)
        
        # find and apply step size
        pk = spec[target] - lastspec # direction
        alphak = (typeof(alpha) == Float64) ? alpha : alpha(k, pk, lastspec)
        spec[target] = lastspec + alphak * pk
        
        chi2s = chi2s(normalizepdf(lastspec),
                           normalizepdf(spec[target])) # Chi Square distance between iterations
        
        # info("DSEA iteration $k/$maxiter ",
        #      fixweighting || smoothing != :none ? "(" : "",
        #      fixweighting ? "fixed weighting" : "",
        #      fixweighting && smoothing != :none ? ", " : "",
        #      smoothing != :none ? string(smoothing) * " smoothing" : "",
        #      fixweighting || smoothing != :none ? ") " : "",
        #      "uses alpha = $alphak (chi2s = $chi2s)")
        
        # optionally monitor progress
        if (inspect != nothing)
            inspect(k, alphak, chi2s, spec)
        end
        
        # stop when convergence is assumed
        if chi2s < epsilon # also holds when alpha is zero
            # info("DSEA convergence assumed from chi2s = $chi2s < epsilon = $epsilon")
            break
        end
        
        # reweighting of items
        if (k < maxiter) # only done if there is a next iteration
            # apply smoothing as intermediate step
            spec[target] = smoothpdf(spec[target], smoothing; kwargs...)
            
            binweights = normalizepdf(  fixweighting ? spec[target] ./ wfactor : spec[target]  )
            traindf[w] = max.([ binweights[findfirst(spec[:level] .== t)] for t in traindf[target] ], 1/size(traindf, 1))
        end
        
    end
    
    # return results
    if returncontributions
        return spec, preddata
    else # default
        return spec
    end
    
end

# spectral reconstruction: sum of confidences in each bin
function _dsea_reconstruct{T<:Number}(preddata::DataFrame, ylevels::AbstractArray{T, 1})
    bincontent = (bin::Float64) -> begin # obtain the column of the bin
        col = Symbol(bin)
        if in(col, names(preddata))
            sum(preddata[col])
        else
            0.0
        end
    end
    return convert(Array{Float64, 1}, map(bincontent, ylevels))
end

