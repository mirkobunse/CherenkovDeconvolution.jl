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
"""
function ibu{T<:Int}(x_data::AbstractArray{T, 1},
                     x_train::AbstractArray{T, 1},
                     y_train::AbstractArray{T, 1},
                     bins_y::AbstractArray{T, 1} = 1:maximum(y_train);
                     kwargs...)
    # recode labels
    y_train, recode_dict = _recode_labels(y_train, bins_y)
    kwargs_dict = Dict(kwargs)
    if haskey(kwargs_dict, :f_0)
        kwargs_dict[:f_0] = _check_prior(kwargs_dict[:f_0], recode_dict)
    end
    
    # deconvolve
    f = ibu(Util.fit_R(y_train, x_train), Util.fit_pdf(x_data, unique(x_train)); kwargs_dict...)
    return _recode_result(f, recode_dict) # revert recoding of labels
end

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
- `inspect = nothing`
  is a function `(k::Int, chi2s::Float64, f_k::Array) -> Any` optionally called in every
  iteration.
- `loggingstream = DevNull`
  is an optional `IO` stream to write log messages to.
"""
function ibu{T<:Number}(R::Matrix{Float64}, g::Array{T, 1};
                        f_0::Array{Float64, 1} = Float64[],
                        smoothing::Function = Base.identity,
                        K::Int = 3,
                        epsilon::Float64 = 0.0,
                        inspect::Function = (args...) -> nothing,
                        loggingstream::IO = DevNull)
    
    # check arguments
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    end
    f_0 = _check_prior(f_0, size(R, 2))
    
    # initial estimate
    f = Util.normalizepdf(f_0)
    inspect(0, NaN, f) # inspect prior
    
    # iterative Bayesian deconvolution
    for k in 1:K
        f_prev = f
        
        # === apply Bayes' rule ===
        f = Util.normalizepdf(_ibu_reverse_transfer(R, f_prev) * g)
        # = = = = = = = = = = = = =
        
        # monitor progress
        chi2s = Util.chi2s(f_prev, f, false) # Chi Square distance between iterations
        info(loggingstream, "IBU iteration $k/$K (chi2s = $chi2s)")
        inspect(k, chi2s, f)
        
        # stop when convergence is assumed
        if chi2s < epsilon
            info(loggingstream, "IBU convergence assumed from chi2s = $chi2s < epsilon = $epsilon")
            break
        end
        
        # == smoothing in between iterations ==
        if k < K
            f = smoothing(f)
        end
        # = = = = = = = = = = = = = = = = = = =
        
    end
    
    return f # return last estimate
    
end

# reverse the transfer with Bayes' rule, given the transfer matrix R and the prior f_0
function _ibu_reverse_transfer(R::Matrix{Float64}, f_0::Array{Float64, 1})
    B = zeros(R')
    for j in 1:size(R, 1)
        B[:, j] = R[j, :] .* f_0 ./ dot(R[j, :], f_0)
    end
    return B
end

