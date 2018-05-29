"""
    ibu(data, train, y, x; kwargs...)

Iterative Bayesian Unfolding of the `y` distribution in the DataFrame `data`.

Given is the observable column `x` and in the `train` DataFrame also the target column `y`.
This method wraps `ibu(R, g)` constructing `R` and `g` from the events in the DataFrames.

### Additional keyword arguments

- `ylevels` optionally specifies the discrete levels of `y`
- `xlevels` optionally specifies the discrete levels of `x`
"""
function ibu{T1 <: Number, T2 <: Number}(data::DataFrame, train::DataFrame,
                                         y::Symbol, x::Symbol;
                                         ylevels::AbstractArray{T1, 1} = sort(unique(train[y])),
                                         xlevels::AbstractArray{T2, 1} = sort(unique(train[x])),
                                         kwargs...)
    
    # estimate response and observable distribution
    R = Util.empiricaltransfer(train[y], train[x], xlevels = xlevels, ylevels = ylevels)
    g = Util.histogram(data[x], xlevels)
    
    return ibu(R, g; kwargs...)
    
end

"""
    ibu(R, g; kwargs...)

Iterative Bayesian Unfolding with the detector response function `R` and the observable
distribution `g`.

### Keyword arguments

- `f_0 = [1/m, .., 1/m]` is the prior target distribution (default is uniform with `m` bins).
- `K = 3` is the maximum number of iterations.
- `epsilon::Float64 = 0` is the minimum symmetric Chi Square distance between iterations.
- `smoothing::Symbol = :none` can also be set to `:polynomial` to apply polynomial smoothing
  [dagostini2010improved], or any other `method` accepted by `smoothpdf`.
  The operation is neither applied to the initial prior, nor to the final result,
  and not to any of the spectra fed into `inspect`.
- `inspect` is an optional function `(k::Int, chi2s::Float, f_k::DataFrame) -> Any` called
  in every iteration `k`.

Any other keyword argument is forwarded to the smoothing operation (if smoothing is applied).
"""
function ibu{T<:Number}(R::AbstractArray{Float64, 2}, g::AbstractArray{T, 1};
                        f_0::AbstractArray{Float64, 1} = ones(size(R, 2)) ./ size(R, 2),
                        K::Int = 3, epsilon::Float64 = 0.0, smoothing::Symbol = :none,
                        inspect::Function = (k, f) -> nothing,
                        kwargs...)
    
    N   = sum(g) # number of examples
    f_0 = Util.normalizepdf(f_0)
    inspect(0, NaN, f_0 .* N) # inspect prior
    
    # iterative Bayesian deconvolution
    for k in 1:K
        
        # apply Bayes' rule, compute the X^2 distance, and update the prior
        f_k   = Util.normalizepdf(_ibu_reverse_transfer(R, f_0) * g)
        chi2s = Util.chi2s(f_0, f_k, false)
        f_0   = f_k
        inspect(k, chi2s, f_0 .* N)
        
        # stop when convergence is assumed
        if chi2s < epsilon
            # info("IBU convergence assumed from chi2s = $chi2s < epsilon = $epsilon")
            break
        end
        
        # smoothing is an intermediate step not performed on the last iteration
        if k < K
            f_0 = Util.smoothpdf(f_0, smoothing; kwargs...)
        end
        
    end
    return f_0 .* N # return last estimate
    
end

# reverse the transfer with Bayes' rule, given the transfer matrix R and the prior f_0
function _ibu_reverse_transfer{T <: Number}(R::Array{Float64, 2}, f_0::AbstractArray{T, 1})
    B = zeros(R')
    for j in 1:size(R, 1)
        B[:, j] = R[j, :] .* f_0 ./ dot(R[j, :], f_0)
    end
    return B
end

