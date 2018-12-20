"""
    ibu(data, train, x, y[, bins_y; kwargs...])

Iterative Bayesian Unfolding of the target distribution in the DataFrame `data`. The
deconvolution is inferred from the DataFrame `train`, where the target column `y` and the
observable column `x` are given.

This function wraps `ibu(R, g; kwargs...)`, constructing `R` and `g` from the examples in
the two DataFrames.
"""
ibu(data::AbstractDataFrame, train::AbstractDataFrame, x::Symbol, y::Symbol,
    bins_y::AbstractArray = 1:maximum(train[y]); kwargs...) =
  ibu(data[x], train[x], train[y], bins_y; kwargs...)

"""
    ibu(x_data, x_train, y_train[, bins_y; kwargs...])

Iterative Bayesian Unfolding of the target distribution, given the observations in the
one-dimensional array `x_data`.

The deconvolution is inferred from `x_train` and `y_train`. Both of these arrays have to be
discrete, i.e., they must contain indices instead of actual values. All expected label
indices (for cases where `y_train` may not contain some of the indices) are optionally
provided as `bins_y`.

This function wraps `ibu(R, g; kwargs...)`, constructing `R` and `g` from the examples in
the three arrays.

**Caution:** In this form, the keyword argument `f_0` always specifies a pdf prior,
irrespective of the value of `fit_ratios`.
"""
ibu( x_data  :: AbstractArray{T, 1},
     x_train :: AbstractArray{T, 1},
     y_train :: AbstractArray{T, 1},
     bins_y  :: AbstractArray{T, 1} = 1:maximum(y_train);
     kwargs... ) where T<:Int =
  _discrete_deconvolution(ibu, x_data, x_train, y_train, bins_y, Dict{Symbol, Any}(kwargs))

"""
    ibu(R, g; kwargs...)

Iterative Bayesian Unfolding with the detector response matrix `R` and the observable
density function `g`.

**Keyword arguments**

- `f_0 = ones(m) ./ m`
  defines the prior, which is uniform by default.
- `smoothing = Base.identity`
  is a function that optionally applies smoothing in between iterations. The operation is
  neither applied to the initial prior, nor to the final result. The function `inspect` is
  called before the smoothing is performed.
- `K = 3`
  is the maximum number of iterations.
- `epsilon = 0.0`
  is the minimum symmetric Chi Square distance between iterations. If the actual distance is
  below this threshold, convergence is assumed and the algorithm stops.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.
  According to this setting, `f_0` specifies a ratio prior or a pdf prior.
- `inspect = nothing`
  is a function `(f_k::Array, k::Int, chi2s::Float64) -> Any` optionally called in every
  iteration.
- `loggingstream = DevNull`
  is an optional `IO` stream to write log messages to.
"""
function ibu(R::Matrix{TR}, g::Vector{Tg};
             f_0::Vector{Float64} = Float64[],
             smoothing::Function = Base.identity,
             K::Int = 3,
             epsilon::Float64 = 0.0,
             fit_ratios::Bool = false,
             inspect::Function = (args...) -> nothing,
             loggingstream::IO = DevNull) where {TR<:Number, Tg<:Number}
    
    # check arguments
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    end
    
    # initial estimate
    f = _check_prior(f_0, size(R, 2), !fit_ratios) # do not normalize if ratios are fitted
    inspect(f, 0, NaN) # inspect prior
    
    # iterative Bayesian deconvolution
    for k in 1:K
        
        # == smoothing in between iterations ==
        f_prev_smooth = k > 1 ? smoothing(f) : f # do not smooth the initial estimate
        f_prev = f # unsmoothed estimate required for convergence check
        # = = = = = = = = = = = = = = = = = = =
        
        # === apply Bayes' rule ===
        f = _ibu_reverse_transfer(R, f_prev_smooth) * g
        if !fit_ratios
            f = Util.normalizepdf(f, warn=false)
        end
        # = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = Util.chi2s(f_prev, f, false) # Chi Square distance between iterations
        info(loggingstream, "IBU iteration $k/$K (chi2s = $chi2s)")
        inspect(f, k, chi2s)
        
        # stop when convergence is assumed
        if chi2s < epsilon
            info(loggingstream, "IBU convergence assumed from chi2s = $chi2s < epsilon = $epsilon")
            break
        end
        
    end
    
    return f # return last estimate
    
end

# reverse the transfer with Bayes' rule, given the transfer matrix R and the prior f_0
function _ibu_reverse_transfer(R::Matrix{T}, f_0::Vector{Float64}) where T<:Number
    B = zeros(Float64, size(R'))
    for j in 1:size(R, 1)
        B[:, j] = R[j, :] .* f_0 ./ dot(R[j, :], f_0)
    end
    return B
end

