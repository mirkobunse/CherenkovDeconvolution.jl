# 
# CherenkovDeconvolution.jl
# Copyright 2018-2021 Mirko Bunse
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
export RUN

"""
    RUN(binning; kwargs...)

The *Regularized Unfolding* method, using a `binning` to discretize the observable features.

**Keyword arguments**

- `n_df = size(R, 2)`
  is the effective number of degrees of freedom. The default `n_df` results in no
  regularization (there is one degree of freedom for each dimension in the result).
- `K = 100`
  is the maximum number of iterations.
- `epsilon = 1e-6`
  is the minimum difference in the loss function between iterations. RUN stops when the
  absolute loss difference drops below `epsilon`.
- `acceptance_correction = nothing` 
  is a tuple of functions (ac(d), inv_ac(d)) representing the acceptance correction
  ac and its inverse operation inv_ac for a data set d.
- `ac_regularisation = true` 
  decides whether acceptance correction is taken into account for regularisation.
  Requires `acceptance_correction` != nothing.
- `log_constant = 1/18394`
  is a selectable constant used in log regularisation to prevent the undefined case log(0).
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, ldiff::Float64, tau::Float64) -> Any` optionally
  called in every iteration.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.
"""
struct RUN <: DiscreteMethod
    binning :: Binning
    acceptance_correction :: Union{Tuple{Function, Function}, Nothing}
    ac_regularisation :: Bool
    epsilon :: Float64
    fit_ratios :: Bool
    inspect :: Function
    K :: Int
    log_constant :: Float64
    n_bins_y :: Int
    n_df :: Int
    RUN(binning;
        acceptance_correction :: Union{Tuple{Function, Function}, Nothing} = nothing,
        ac_regularisation :: Bool     = true,
        epsilon           :: Float64  = 1e-6,
        fit_ratios        :: Bool     = false,
        inspect           :: Function = (args...) -> nothing,
        K                 :: Int64    = 100,
        log_constant      :: Float64  = 1/18394,
        n_bins_y          :: Int      = -1,
        n_df              :: Int      = typemax(Int)
    ) = new(binning, acceptance_correction, ac_regularisation, epsilon, fit_ratios, inspect, K, log_constant, n_bins_y, n_df)
end

binning(run::RUN) = run.binning
expects_normalized_R(run::RUN) = !run.fit_ratios
expects_normalized_g(run::RUN) = false
expected_n_bins_y(run::RUN) = run.n_bins_y

run( x_data  :: AbstractVector{T},
     x_train :: AbstractVector{T},
     y_train :: AbstractVector{T},
     bins_y  :: AbstractVector{T} = 1:maximum(y_train);
     kwargs... ) where T<:Int =
  error("No deprecation redirection implemented")

function deconvolve(
        run::RUN,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number}

    # limit the unfolding to non-zero bins
    if any(g .<= 0)
        nonzero = g .> 0
        @warn "Limiting RUN to $(sum(nonzero)) of $(length(g)) observeable non-zero bins"
        g = g[nonzero]
        R = R[nonzero, :]
    end

    # check arguments
    check_discrete_arguments(R, g)
    m = size(R, 2) # dimension of f
    if m > size(R, 1)
        @warn "RUN is performed on more target than observable bins - results may be unsatisfactory"
    end

    # set up the loss function
    l   = _maxl_l(R, g) # the objective function,
    g_l = _maxl_g(R, g) # ..its gradient,
    H_l = _maxl_H(R, g) # ..and its Hessian
    C   = _tikhonov_binning(m) # the Tikhonov matrix (not in l and its derivatives)

    # set up acceptance correction
    ac_regularisation = run.ac_regularisation
    a = nothing # the default
    if run.acceptance_correction !== nothing
        ac, inv_ac = run.acceptance_correction
        if ac_regularisation
            a = inv_ac(ones(m))
        end
    elseif ac_regularisation
        @warn "Performing acceptance correction regularisation requires a given acceptance_correction object"
        ac_regularisation = false
    end

    # the initial estimate is the zero vector
    f = zeros(Float64, m)

    # the first iteration is a least squares fit
    H_lsq = _lsq_H(R, g)(f)
    if !all(isfinite.(H_lsq))
        @warn "LSq hessian contains Infs or NaNs - replacing these by zero"
        H_lsq[.!(isfinite.(H_lsq))] .= 0.0
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
    f_inspect = f
    if run.fit_ratios
        f_inspect = f_inspect .* f_trn # convert a ratio solution to a pdf solution
    end
    run.inspect(DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f_inspect), warn=false), 1, NaN, NaN)

    # subsequent iterations maximize the likelihood
    l_prev = l(f) # loss from the previous iteration
    for k in 2:run.K

        # gradient and Hessian at the last estimate
        g_f = g_l(f)
        H_f = H_l(f)
        if !all(isfinite.(H_f))
            @warn "MaxL hessian contains Infs or NaNs - replacing these by zero"
            H_f[.!(isfinite.(H_f))] .= 0.0
        end

        # eigendecomposition of the Hessian: H_f == U*D*U' (complex conversion required if more y than x bins)
        eigen_H = eigen(H_f)
        U = eigen_H.vectors
        D = Matrix(Diagonal(real.(complex.(eigen_H.values) .^ (-1/2)))) # D^(-1/2)

        # eigendecomposition of transformed Tikhonov matrix: C2 == U_C*S*U_C'
        eigen_C = eigen(Symmetric( D*U' * C * U*D ))

        # select tau (special case: no regularization if n_df == m)
        tau = run.n_df < m ? _tau(run.n_df, eigen_C.values) : 0.0

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
        g_f += _C_g(tau, C; a=a, log_constant=run.log_constant)(f) # regularized gradient
        H_f += _C_H(tau, C; a=a, log_constant=run.log_constant)(f) # regularized Hessian
        f += try
            - inv(H_f) * g_f
        catch err
            if isa(err, SingularException) || isa(err, LAPACKException)
                try # try again with pseudo inverse
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
        l_now = l(f) + _C_l(tau, C; a=a, ac_regularisation=ac_regularisation, log_constant=run.log_constant)(f)
        ldiff = l_prev - l_now
        @debug "RUN iteration $k/$(run.K) uses tau = $tau (ldiff = $ldiff)"
        f_inspect = f
        if run.fit_ratios
            f_inspect = f_inspect .* f_trn # convert a ratio solution to a pdf solution
        end
        run.inspect(DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f_inspect), warn=false), k, ldiff, tau)

        # stop when convergence is assumed
        if abs(ldiff) < run.epsilon
            @debug "RUN convergence assumed from ldiff = $ldiff < epsilon = $(run.epsilon)"
            break
        end
        l_prev = l_now

    end

    if run.fit_ratios
        f = f .* f_trn # convert a ratio solution to a pdf solution
    end
    return DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f))
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
        if fj > 0.0
            fj - g[j]*log(fj)
        else 
            -Inf
        end
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
_C_l(tau::Float64, C::Matrix{Float64};
     a::Union{Nothing, Vector{Float64}}=nothing, ac_regularisation::Bool=true, log_constant::Float64=1/18394) =
    f̄ -> begin   
        if ac_regularisation && a !== nothing 
            f̄_a_log = map(x -> x<=0.0 ? 0.0 : log(x), a .* f̄)
            tau/2 * dot(f̄_a_log, C * f̄_a_log)
        else
            tau/2 * dot(f̄, C*f̄)
        end
    end

# regularization term in gradient of objective
_C_g(tau::Float64, C::Matrix{Float64};
    a::Union{Nothing, Vector{Float64}}=nothing, ac_regularisation::Bool=true, log_constant::Float64=1/18394) =
    f̄ -> begin
        if ac_regularisation && a !== nothing 
            f̄_d = map(x -> x<=0.0 ? 0.0 : 1/x, f̄)
            F = Diagonal(f̄_d)
            f̄_a_log = map(x -> x<=0.0 ? 0.0 : log(x), a .* f̄)
            C_g = tau * F * C * f̄_a_log
        else
            tau * C * f̄
        end
    end

# regularization term in the Hessian of objective
_C_H(tau::Float64, C::Matrix{Float64};
    a::Union{Nothing, Vector{Float64}}=nothing, ac_regularisation::Bool=true, log_constant::Float64=1/18394) =
    f̄ -> begin
        if ac_regularisation && a !== nothing 
            f̄ = max.(f̄, 0.0)
            H = tau .* C ./ (f̄ * transpose(f̄))
            f̄_a_log = map(x -> x<=0.0 ? 0.0 : log(x), a .* f̄)
            H[diagind(H)] .= diag( - tau .* C * (repeat(f̄_a_log, 1, length(f̄)) - diagm(ones(length(f̄)))) ./ f̄ .^2)
            if !all(isfinite.(H))
                @warn "regularized hessian contains Infs or NaNs - replacing these by zero"
                H[.!(isfinite.(H))] .= 0.0
            end
            return H
        else        
            tau * C
        end
    end

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
