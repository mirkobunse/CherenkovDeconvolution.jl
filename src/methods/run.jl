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
"""
    run(data, train, x, y[, bins_y]; kwargs...)

    run(x_data, x_train, y_train[, bins_y]; kwargs...)

    run(R, g; kwargs...)


Deconvolve the observed data applying the *Regularized Unfolding* trained on the given
training set.

The vectors `x_data`, `x_train`, and `y_train` (or accordingly `data[x]`, `train[x]`, and
`train[y]`) must contain label/observation indices rather than actual values. All expected
indices in `y_train` are optionally provided as `bins_y`. Alternatively, the detector
response matrix `R` and the observed density vector `g` can be given directly.


**Keyword arguments**

- `n_df = size(R, 2)`
  is the effective number of degrees of freedom. The default `n_df` results in no
  regularization (there is one degree of freedom for each dimension in the result).
- `K = 100`
  is the maximum number of iterations.
- `epsilon = 1e-6`
  is the minimum difference in the loss function between iterations. RUN stops when the
  absolute loss difference drops below `epsilon`.
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, ldiff::Float64, tau::Float64) -> Any` optionally
  called in every iteration.
- `loggingstream = devnull`
  is an optional `IO` stream to write log messages to.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.


**Caution:** According to the value of `fit_ratios`, the keyword argument `f_0` specifies a
ratio prior or a pdf prior, but only in the third form. In the other forms, `f_0` always
specifies a pdf prior.
"""
run( data   :: AbstractDataFrame,
     train  :: AbstractDataFrame,
     x      :: Symbol,
     y      :: Symbol,
     bins_y :: AbstractVector = 1:maximum(train[y]);
     kwargs... ) =
  run(data[x], train[x], train[y], bins_y; kwargs...) # DataFrame form


# Vector form
run( x_data  :: AbstractVector{T},
     x_train :: AbstractVector{T},
     y_train :: AbstractVector{T},
     bins_y  :: AbstractVector{T} = 1:maximum(y_train);
     kwargs... ) where T<:Int =
  _discrete_deconvolution(run, x_data, x_train, y_train, bins_y, Dict{Symbol, Any}(kwargs), normalize_g=false)


function run( R :: Matrix{TR},
	          g :: Vector{Tg};
              n_df    :: Number   = size(R, 2),
              K       :: Int      = 100,
              epsilon :: Float64  = 1e-6,
              inspect :: Function = (args...) -> nothing,
              loggingstream :: IO = devnull,
              kwargs... ) where {TR<:Number, Tg<:Number}
    
    if any(g .<= 0) # limit unfolding to non-zero bins
        nonzero = g .> 0
        @warn "Limiting RUN to $(sum(nonzero)) of $(length(g)) observeable non-zero bins"
        g = g[nonzero]
        R = R[nonzero, :]
    end
    
    # check arguments
    m = size(R, 2) # dimension of f
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    end
    if m > size(R, 1)
        @warn "RUN is performed on more target than observable bins - results may be unsatisfactory"
    end
    
    # set up the loss function
    l   = _maxl_l(R, g) # the objective function,
    g_l = _maxl_g(R, g) # ..its gradient,
    H_l = _maxl_H(R, g) # ..and its Hessian
    C   = _tikhonov_binning(m) # the Tikhonov matrix (not in l and its derivatives)
    
    # initial estimate is the zero vector
    f = zeros(Float64, m)
    
    # the first iteration is a least squares fit
    H_lsq = _lsq_H(R, g)(f)
    if !all(isfinite.(H_lsq))
        @warn "LSq hessian contains Infs or NaNs - replacing these by zero"
        H_lsq[.!(isfinite.(H_lsq))] = 0.0
    end
    f += try
        - inv(H_lsq) * _lsq_g(R, g)(f)
    catch err
        if isa(err, SingularException) # pinv instead of inv only required if more y than x bins
            @warn "LSq hessian is singular - using pseudo inverse in RUN"
            - pinv(H_lsq) * _lsq_g(R, g)(f)
        else
            rethrow(err)
        end
    end
    inspect(f, 1, NaN, NaN)
    
    # subsequent iterations maximize the likelihood
    l_prev = l(f) # loss from the previous iteration
    for k in 2:K
        
        # gradient and Hessian at the last estimate
        g_f = g_l(f)
        H_f = H_l(f)
        if !all(isfinite.(H_f))
            @warn "MaxL hessian contains Infs or NaNs - replacing these by zero"
            H_f[.!(isfinite.(H_f))] = 0.0
        end
        
        # eigendecomposition of the Hessian: H_f == U*D*U' (complex conversion required if more y than x bins)
        eigen_H = eigen(H_f)
        U = eigen_H.vectors
        D = Matrix(Diagonal(real.(complex.(eigen_H.values) .^ (-1/2)))) # D^(-1/2)
        
        # eigendecomposition of transformed Tikhonov matrix: C2 == U_C*S*U_C'
        eigen_C = eigen(Symmetric( D*U' * C * U*D ))
        
        # select tau (special case: no regularization if n_df == m)
        tau = n_df < m ? _tau(n_df, eigen_C.values) : 0.0
        
        # 
        # Taking a step in the transformed problem and transforming back to the actual
        # solution is numerically difficult because the eigendecomposition introduces some
        # error. In the transformed problem, therefore only tau is chosen. The step is taken 
        # in the original problem instead of the commented-out solution.
        # 
        # U_C = eigen_C.vectors
        # S   = diagm(eigen_C.values)
        # f_2 = 1/2 * inv(eye(S) + tau*S) * (U*D*U_C)' * (H_f * f - g_f)
        # f   = (U*D*U_C) * f_2
        # 
        g_f += _C_g(tau, C)(f) # regularized gradient
        H_f += _C_H(tau, C)(f) # regularized Hessian
        f += try
            - inv(H_f) * g_f
        catch err
            
            # try again with pseudo inverse
            if isa(err, SingularException) || isa(err, LAPACKException)
                try
                    step = - pinv(H_f) * g_f # update step
                    if isa(err, SingularException)
                        @warn "MaxL Hessian is singular - using pseudo inverse in RUN"
                    else
                        @warn "LAPACKException on inversion of MaxL Hessian - using pseudo inverse in RUN"
                    end
                    step # return update step after warning is emitted and only if computation is successful
                catch err2
                    if isa(err, LAPACKException) # same exception occurs with pinv?
                        @warn "LAPACKException on pseudo inversion of MaxL Hessian - not performing an update in RUN"
                        zero(f) # return zero step to not update f
                    else
                        rethrow(err2)
                    end
                end
            else
                rethrow(err)
            end
            
        end
        
        # monitor progress
        l_now = l(f) + _C_l(tau, C)(f)
        ldiff = l_prev - l_now
        @debug "RUN iteration $k/$K uses tau = $tau (ldiff = $ldiff)"
        inspect(f, k, ldiff, tau)
        
        # stop when convergence is assumed
        if abs(ldiff) < epsilon
            @debug "RUN convergence assumed from ldiff = $ldiff < epsilon = $epsilon"
            break
        end
        l_prev = l_now
        
    end
    return f
    
end

# Brute-force search of a tau satisfying the n_df relation
function _tau(n_df::Number, eigvals_C::Vector{Float64})
    res = _tau(n_df, tau -> sum([ 1/(1 + tau*v) for v in eigvals_C ])) # recursive subroutine
    return res[1] # final tau
end

# Recursive subroutine of _tau(::Number, ::Vector{Float64}) for higher precision
function _tau(n_df::Number, taufunction::Function, min::Float64=-.01, max::Float64=-18.0, i::Int64=2)
    taus = 10 .^ range(min, stop=max, length=1000)
    ndfs = map(taufunction, taus)
    
    best = findmin(abs.(ndfs .- n_df)) # tuple from difference and index of minimum
    tau  = taus[best[2]]
    ndf  = ndfs[best[2]]
    diff = best[1]
    
    if i == 1
        return tau, ndf, [ diff ] # recursive anchor
    else
        # search more closely around best fit
        max = log10(taus[best[2] < length(taus) ? best[2] + 1 : best[2]]) # upper bound of subsequent search
        min = log10(taus[best[2] > 1            ? best[2] - 1 : best[2]]) # lower bound
        subsequent = _tau(n_df, taufunction, min, max, i-1) # result of recursion
        return subsequent[1], subsequent[2], vcat(diff, subsequent[3])
    end
end


# objective function: negative log-likelihood
_maxl_l(R::Matrix{TR}, g::AbstractVector{Tg}) where {TR<:Number, Tg<:Number} =
    f -> sum(begin
        fj = dot(R[j,:], f)
        fj - g[j]*real(log(complex(fj)))
    end for j in 1:length(g))

# gradient of objective
_maxl_g(R::Matrix{TR}, g::AbstractVector{Tg}) where {TR<:Number, Tg<:Number} =
    f -> [ sum([ R[j,i] - g[j]*R[j,i] / dot(R[j,:], f) for j in 1:length(g) ])
           for i in 1:length(f) ]

# hessian of objective
_maxl_H(R::Matrix{TR}, g::AbstractVector{Tg}) where {TR<:Number, Tg<:Number} =
    f -> begin
        res = zeros(Float64, (length(f), length(f)))
        for i1 in 1:length(f), i2 in 1:length(f)
            res[i1,i2] = sum( # hessian in cell (i1,i2)
                g[j]*R[j,i1]*R[j,i2] / dot(R[j,:], f)^2
            for j in 1:length(g))
        end
        Matrix(Symmetric(res)) # Matrix constructor converts symmetric result to a full matrix
    end


# objective function: least squares
_lsq_l(R::Matrix{TR}, g::AbstractVector{Tg}) where {TR<:Number, Tg<:Number} =
    f -> sum([ (g[j] - dot(R[j,:], f))^2 / g[j] for j in 1:length(g) ])/2

# gradient of objective
_lsq_g(R::Matrix{TR}, g::AbstractVector{Tg}) where {TR<:Number, Tg<:Number} =
    f -> [ sum([ -R[j,i] * (g[j] - dot(R[j,:], f)) / g[j] for j in 1:length(g) ])
           for i in 1:length(f) ]

# hessian of objective
_lsq_H(R::Matrix{TR}, g::AbstractVector{Tg}) where {TR<:Number, Tg<:Number} =
    f -> begin
        res = zeros(Float64, (length(f), length(f)))
        for i1 in 1:length(f), i2 in 1:length(f)
            res[i1,i2] = sum( # hessian in cell (i1,i2)
                R[j,i1]*R[j,i2] / g[j]
            for j in 1:length(g))
        end
        Matrix(Symmetric(res))
    end

# regularization term in objective function (both LSq and MaxL)
_C_l(tau::Float64, C::Matrix{Float64}) = f -> tau/2 * dot(f, C*f)

# regularization term in gradient of objective
_C_g(tau::Float64, C::Matrix{Float64}) = f -> tau * C * f

# regularization term in the Hessian of objective
_C_H(tau::Float64, C::Matrix{Float64}) = f -> tau * C

# Construct a Tikhonov matrix for binned discretization, as given in [cowan1998statistical, p. 169].
# This is equivalent to the notation in [blobel2002unfolding_long]!
_tikhonov_binning(m::Int) =
    if m < 1
        throw(ArgumentError("m has to be greater than zero"))
    elseif m < 3 # stupid case
        Matrix(1.0I, m, m) # identity matrix (was Base.eye(m) before julia-0.7)
    elseif m == 3 # not quite an intelligent case
        convert(Matrix{Float64},
            diagm( 0 => repeat([1], 3),
                   1 => repeat([-1], 2),
                  -1 => repeat([-1], 2) ))
    else # usual case
        convert(Matrix{Float64},
            diagm( 0 => vcat([1, 5], repeat([6], inner=max(0, m-4)), [5, 1]),
                   1 => vcat([-2], repeat([-4], inner=m-3), [-2]),
                  -1 => vcat([-2], repeat([-4], inner=m-3), [-2]),
                   2 => repeat([1], inner=m-2),
                  -2 => repeat([1], inner=m-2) ))
    end

