"""
    dsea(data, train, y, train_predict[, bins]; kwargs...)

    dsea(X_data, X_train, y_train, train_predict[, bins]; kwargs...)


Deconvolve the observed data with *DSEA/DSEA+* trained on the given training set.

The first form of this function works on the two DataFrames `data` and `train`, where `y`
specifies the target column to be deconvolved - this column has to be present in the
DataFrame `train`. The second form works on vectors and matrices.

To facilitate classification, `y_train` (or `train[y]` in the first form) must contain label
indices rather than actual values. All expected indices are optionally provided as `bins`.
The function object `train_predict(X_data, X_train, y_train, w_train) -> Matrix` trains and
applies a classifier, obtaining a confidence matrix. All of its arguments but `w_train`,
which is updated in each iteration, are simply passed through from `dsea`.


**Keyword arguments**

- `f_0 = ones(m) ./ m`
  defines the prior, which is uniform by default
- `fixweighting = true`
  sets, whether or not the weight update fix is applied. This fix is proposed in my Master's
  thesis and in the corresponding paper.
- `alpha = 1.0`
  is the step size taken in every iteration.
  This parameter can be either a constant value or a function with the signature
  `(k::Int, pk::AbstractVector{Float64}, f_prev::AbstractVector{Float64} -> Float`,
  where `f_prev` is the estimate of the previous iteration and `pk` is the direction that
  DSEA takes in the current iteration `k`.
- `smoothing = Base.identity`
  is a function that optionally applies smoothing in between iterations
- `K = 1`
  is the maximum number of iterations.
- `epsilon = 0.0`
  is the minimum symmetric Chi Square distance between iterations. If the actual distance is
  below this threshold, convergence is assumed and the algorithm stops.
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, chi2s::Float64, alpha::Float64) -> Any` optionally
  called in every iteration.
- `loggingstream = DevNull`
  is an optional `IO` stream to write log messages to.
- `return_contributions = false`
  sets, whether or not the contributions of individual examples in `X_data` are returned as
  a tuple together with the deconvolution result.
- `features = setdiff(names(train), [y])`
  specifies which columns in `data` and `train` to be used as features - only applicable to
  the first form of this function.
"""
dsea( data          :: AbstractDataFrame,
      train         :: AbstractDataFrame,
      y             :: Symbol,
      train_predict :: Function,
      bins          :: AbstractVector{T} = 1:maximum(train[y]);
      features      :: AbstractVector{Symbol} = setdiff(names(train), [y]),
      kwargs... ) where T<:Int =
  dsea(Util.df2X(data, features),
       Util.df2Xy(train, y, features)...,
       train_predict,
       bins;
       kwargs...) # DataFrame form


# Vector/Matrix form
# 
# Here, X_data, X_train, and y_train are only converted to actual Array objects because
# ScikitLearn.jl goes mad when some of the other sub-types of AbstractArray are used. The
# actual implementation is below.
dsea( X_data        :: AbstractMatrix{TN},
      X_train       :: AbstractMatrix{TN},
      y_train       :: AbstractVector{TI},
      train_predict :: Function,
      bins          :: AbstractVector{TI} = 1:maximum(y_train);
      kwargs... ) where {TN<:Number, TI<:Int} =
  _dsea(convert(Matrix, X_data), convert(Matrix, X_train), convert(Vector, y_train),
        train_predict, convert(Vector, bins); kwargs...)


function _dsea(X_data::Matrix{TN},
               X_train::Matrix{TN},
               y_train::Vector{TI},
               train_predict::Function,
               bins::Vector{TI} = 1:maximum(y_train);
               f_0::Vector{Float64} = Float64[],
               fixweighting::Bool = true,
               alpha::Union{Float64, Function} = 1.0,
               smoothing::Function = Base.identity,
               K::Int64 = 1,
               epsilon::Float64 = 0.0,
               inspect::Function = (args...) -> nothing,
               loggingstream::IO = DevNull,
               return_contributions::Bool = false) where {TN<:Number, TI<:Int}
    
    # recode labels and check arguments
    recode_dict, y_train = _recode_indices(bins, y_train)
    if size(X_data, 2) != size(X_train, 2)
        throw(ArgumentError("X_data and X_train do not have the same number of features"))
    end
    f_0 = _check_prior(f_0, recode_dict)
    
    # initial estimate
    f       = f_0
    f_train = Util.fit_pdf(y_train, laplace=true)                            # training pdf with Laplace correction
    w_bin   = fixweighting ? Util.normalizepdf(f ./ f_train, warn=false) : f # bin weights
    w_train = _dsea_weights(y_train, w_bin)                                  # instance weights
    inspect(_recode_result(f, recode_dict), 0, NaN, NaN)
    
    # iterative deconvolution
    proba = Matrix{Float64}(0, 0) # empty matrix
    for k in 1:K
        f_prev = f
        
        # === update the estimate ===
        proba     = train_predict(X_data, X_train, y_train, w_train)
        f_next    = _dsea_reconstruct(proba) # original DSEA reconstruction
        f, alphak = _dsea_step( k,
                                _recode_result(f_next, recode_dict),
                                _recode_result(f_prev, recode_dict),
                                alpha ) # step size function assumes original coding
        f = _check_prior(f, recode_dict) # re-code result of _dsea_step
        # = = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = Util.chi2s(f_prev, f, false) # Chi Square distance between iterations
        info(loggingstream, "DSEA iteration $k/$K uses alpha = $alphak (chi2s = $chi2s)")
        inspect(_recode_result(f, recode_dict), k, chi2s, alphak)
        
        # stop when convergence is assumed
        if chi2s < epsilon # also holds when alpha is zero
            info(loggingstream, "DSEA convergence assumed from chi2s = $chi2s < epsilon = $epsilon")
            break
        end
        
        # == smoothing and reweighting in between iterations ==
        if k < K
            f = smoothing(f)
            w_bin   = fixweighting ? Util.normalizepdf(f ./ f_train, warn=false) : f
            w_train = _dsea_weights(y_train, w_bin)
        end
        # = = = = = = = = = = = = = = = = = = = = = = = = = = =
        
    end
    
    f_rec = _recode_result(f, recode_dict) # revert recoding of labels
    return return_contributions ? (f_rec, proba) : f_rec # result may contain contributions
    
    # TODO proba still has to be recoded, too!
    
end

# the weights of training instances are based on the bin weights in w_bin
_dsea_weights(y_train::Vector{T}, w_bin::Vector{Float64}) where T<:Int =
    max.(w_bin[y_train], 1/length(y_train)) # Laplace correction

# the reconstructed estimate is the sum of confidences in each bin
_dsea_reconstruct(proba::Matrix{Float64}) =
    Util.normalizepdf(map(i -> sum(proba[:, i]), 1:size(proba, 2)), warn=false)

# the step taken by DSEA+, where alpha may be a constant or a function
function _dsea_step(k::Int64, f::Vector{Float64}, f_prev::Vector{Float64},
                    alpha::Union{Float64, Function})
    pk     = f - f_prev                                              # search direction
    alphak = typeof(alpha) == Float64 ? alpha : alpha(k, pk, f_prev) # function or float
    return  f_prev + alphak * pk,  alphak                            # return tuple
end


"""
    alpha_decay_exp(eta::Float64, a_1::Float64=1.0)

Return a `Function` object with the signature required by the `alpha` parameter in `dsea`.
This object reduces the `a_1` stepsize taken in iteration 1 by `eta` in each subsequent
iteration:

    alpha = a_1 * eta^(k-1).
"""
alpha_decay_exp(eta::Float64, a_1::Float64=1.0) =
    (k::Int, pk::Vector{Float64}, f::Vector{Float64}) -> a_1 * eta^(k-1)

"""
    alpha_decay_mul(eta::Float64, a_1::Float64=1.0)

Return a `Function` object with the signature required by the `alpha` parameter in `dsea`.
This object reduces the `a_1` stepsize taken in iteration 1 by `eta` in each subsequent
iteration:

    alpha = a_1 * k ^ (eta-1)
    
For example, eta=.5 yields alpha = 1/sqrt(k).
"""
alpha_decay_mul(eta::Float64, a_1::Float64=1.0) =
    (k::Int, pk::Vector{Float64}, f::Vector{Float64}) -> a_1 * k^(eta-1)

"""
    alpha_adaptive_run(x_data, x_train, y_train[, tau = 0]; bins, bins_x)

Return a `Function` object with the signature required by the `alpha` parameter in `dsea`.
This object adapts the DSEA step size to the current estimate by maximizing the likelihood
of the next estimate in the search direction of the current iteration.
"""
function alpha_adaptive_run( x_data  :: Vector{T},
                             x_train :: Vector{T},
                             y_train :: Vector{T},
                             tau     :: Number = 0.0;
                             bins    :: AbstractVector{T} = 1:maximum(y_train),
                             bins_x  :: AbstractVector{T} = 1:maximum(vcat(x_data, x_train)) ) where T<:Int
    # set up the discrete deconvolution problem
    R = Util.fit_R(y_train, x_train, bins_y = bins, bins_x = bins_x)
    g = Util.fit_pdf(x_data, bins_x, normalize = false) # absolute counts instead of pdf
    
    # set up negative log likelihood function to be minimized
    C = _tikhonov_binning(size(R, 2))       # regularization matrix (from run.jl)
    maxl_l = _maxl_l(R, g)                  # function of f (from run.jl)
    maxl_C = _C_l(tau, C)                   # regularization term (from run.jl)
    negloglike = f -> maxl_l(f) + maxl_C(f) # regularized objective function
    
    # return step size function
    return (k::Int, pk::Vector{Float64}, f::Vector{Float64}) -> begin
        a_min, a_max = _alpha_range(pk, f)
        optimize(a -> negloglike(f + a * pk), a_min, a_max).minimizer # from Optim.jl
    end
end

# range of admissible alpha values
function _alpha_range(pk::Vector{Float64}, f::Vector{Float64})
    # find alpha values for which the next estimate would be zero in one dimension
    a_zero = - (f[pk.!=0] ./ pk[pk.!=0]) # ignore zeros in pk, for which alpha is arbitrary
    
    # for positive pk[i] (negative a_zero[i]), alpha has to be larger than a_zero[i]
    # for negative pk[i] (positive a_zero[i]), alpha has to be smaller than a_zero[i]
    a_min = maximum(vcat(a_zero[a_zero .< 0], 0)) # vcat selects a_min = 0 if no pk[i]>0 is present
    a_max = minimum(a_zero[a_zero .>= 0])
    return a_min, a_max
end

