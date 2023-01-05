# 
# CherenkovDeconvolution.jl
# Copyright 2018-2023 Mirko Bunse
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
export PRUN

"""
    PRUN(binning; kwargs...)

A version of the *Regularized Unfolding* method that is constrained to positive results.
Like the original version, it uses a `binning` to discretize the observable features.

**Keyword arguments**

- `tau = 0.0`
  determines the regularisation strength.
- `K = 100`
  is the maximum number of iterations.
- `epsilon = 1e-6`
  is the minimum difference in the loss function between iterations. RUN stops when the
  absolute loss difference drops below `epsilon`.
- `f_0 = ones(size(R, 2))`
  Starting point for the interior-point Newton optimization.
- `acceptance_correction = nothing` 
  is a tuple of functions (ac(d), inv_ac(d)) representing the acceptance correction
  ac and its inverse operation inv_ac for a data set d.
- `ac_regularisation = true` 
  decides whether acceptance correction is taken into account for regularisation.
  Requires `acceptance_correction` != nothing.
- `log_constant = 1/18394`
  is a selectable constant used in log regularisation to prevent the undefined case log(0).
- `inspect = nothing`
  is a function `(f_k::Vector, k::Int, ldiff::Float64) -> Any` called in each iteration.
- `warn = true`
  determines whether warnings about negative values are emitted during normalization.
- `fit_ratios = false` (**discouraged**)
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.
"""
struct PRUN <: DiscreteMethod
    binning :: Binning
    acceptance_correction :: Union{Tuple{Function, Function}, Nothing}
    ac_regularisation :: Bool
    epsilon :: Float64
    f_0 :: Union{Vector{Float64},Nothing}
    fit_ratios :: Bool
    inspect :: Function
    K :: Int
    log_constant :: Float64
    n_bins_y :: Int
    tau :: Float64
    warn :: Bool
    function PRUN(binning :: Binning;
            acceptance_correction :: Union{Tuple{Function, Function}, Nothing} = nothing,
            ac_regularisation :: Bool     = true,
            epsilon           :: Float64  = 1e-6,
            f_0               :: Union{Vector{Float64},Nothing} = nothing,
            fit_ratios        :: Bool     = false,
            inspect           :: Function = (args...) -> nothing,
            K                 :: Int      = 100,
            log_constant      :: Float64  = 1/18394,
            n_bins_y          :: Int      = -1,
            tau               :: Float64  = 0.0,
            warn              :: Bool     = true)
        if fit_ratios
            @warn "fit_ratios = true is an experimental feature that is discouraged for PRUN"
        end
        return new(binning, acceptance_correction, ac_regularisation, epsilon, f_0, fit_ratios, inspect, K, log_constant, n_bins_y, tau, warn)
    end
end

binning(prun::PRUN) = prun.binning
prior(prun::PRUN) = prun.f_0
expects_normalized_R(prun::PRUN) = !prun.fit_ratios
expects_normalized_g(prun::PRUN) = false
expected_n_bins_y(prun::PRUN) = prun.n_bins_y

function deconvolve(
        prun::PRUN,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f},
        f_0::Union{Vector{T_f},Nothing}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number}

    # limit the unfolding to non-zero bins
    if any(g .<= 0)
        nonzero = g .> 0
        @debug "Limiting PRUN to $(sum(nonzero)) of $(length(g)) observeable non-zero bins"
        g = g[nonzero]
        R = R[nonzero, :]
    end

    # check arguments
    check_discrete_arguments(R, g)
    m = size(R, 2) # dimension of f
    if m > size(R, 1)
        @warn "PRUN is performed on more target than observable bins - results may be unsatisfactory"
    end

    # set the optional prior
    if f_0 == nothing
        if prun.fit_ratios # default prior for ratios
            f_0 = ones(length(f_trn))
        else # default uniform prior
            f_0 = ones(length(f_trn)) ./ length(f_trn)
        end
    elseif prun.fit_ratios
        f_0 = f_0 ./ f_trn # convert to a ratio prior
    end

    # set up acceptance correction
    ac_regularisation = prun.ac_regularisation
    a = nothing # the default
    if prun.acceptance_correction !== nothing
        ac, inv_ac = prun.acceptance_correction
        if ac_regularisation
            a = inv_ac(ones(m))
        end
    elseif ac_regularisation
        @debug "Performing acceptance correction regularisation requires a given acceptance_correction object"
        ac_regularisation = false
    end

    # set up regularized loss function
    l = _maxl_l(R, g)
    C = _tikhonov_binning(m)
    C_l = _C_l(prun.tau, C; a=a, ac_regularisation=ac_regularisation, log_constant=prun.log_constant)
    l_reg = f -> l(f) + C_l(f)

    # regularized gradient
    g!(G, x) = begin
        grad_l = _maxl_g(R, g)(x)
        grad_C = _C_g(prun.tau, C; a=a, ac_regularisation=ac_regularisation, log_constant=prun.log_constant)(x)
        grad_reg = grad_l .+ grad_C
        for j=1:m
            G[j] = grad_reg[j]
        end
    end

    # regularized Hessian
    h!(H, x) = begin
        hess_l = _maxl_H(R,g)(x)
        hess_C = _C_H(prun.tau, C; a=a, ac_regularisation=ac_regularisation, log_constant=prun.log_constant)(x)
        hess_reg = hess_l .+ hess_C
        J,K = size(H)
        for j=1:J, k=1:K
            H[j,k] = hess_reg[j,k]
        end
    end

    # interior-point Newton optimization in [0.0, Inf]
    lx = zeros(m)
    ux = fill(Inf, m)
    dfc = TwiceDifferentiableConstraints(lx, ux)
    df = TwiceDifferentiable(l_reg, g!, h!, f_0)

    conf = Optim.Options(
        g_tol = prun.epsilon,
        iterations = prun.K, # maximum number of iterations
        allow_f_increases = true,
        store_trace = true,
        extended_trace = true,
        successive_f_tol = 2
    )
    res = optimize(df, dfc, f_0, IPNewton(), conf)

    # evaluation
    f = [ DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f_k), warn=prun.warn) for f_k in Optim.x_trace(res) ]
    k = Optim.iterations(res)
    epsilon = Optim.g_norm_trace(res)
    prun.inspect.(f, collect(0:k), epsilon) # make up leeway
    return f[k]
end

# deprecated syntax (the IdentityBinning is defined in src/methods/run.jl)
export p_run
function p_run(
        x_obs  :: AbstractVector{T},
        x_trn  :: AbstractVector{T},
        y_trn  :: AbstractVector{T},
        bins_y :: AbstractVector{T} = 1:maximum(y_trn);
        kwargs...
        ) where T<:Int
    Base.depwarn(join([
        "`p_run(data, config)` is deprecated; ",
        "please call `deconvolve(PRUN(config), data)` instead"
    ]), :p_run)
    p_run = PRUN(IdentityBinning(); n_bins_y=length(bins_y), kwargs...)
    return deconvolve(
        p_run,
        reshape(x_obs, (length(x_obs), 1)), # treat as a matrix
        reshape(x_trn, (length(x_trn), 1)),
        y_trn
    )
end
