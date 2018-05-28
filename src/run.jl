"""
    run(data, train, y, x; kwargs...)

RUN deconvolution of the `y` distribution in the DataFrame `data`.

Given is the observable column `x` and in the `train` DataFrame also the target column `y`.
This method wraps `run(R, g, n_df)` constructing `R`, `g`, and `n_df` from its arguments.

### Additional keyword arguments

- `ylevels` optionally specifies the discrete levels of `y`
- `xlevels` optionally specifies the discrete levels of `x`
- `method = :expand` specifies how the discrete deconvolution problem is obtained. By
  default, the `train` and `data` sets are discretized using `ydiscr` and `xdiscr` with
  the `factor` multiple of bins in `y`. If `method = :reduce`, `data` and `train` can be
  discretized beforehand. The fit is afterwards reduced by the `factor`.
- `factor = 1` is the factor of the regularization strength; `n_df = dim(f)` is enforced
  by the `method`.
- `keepdim = true` specifies, if a reduced or expanded result is projected back to the
  original dimensionality.
- `ydiscr::Discretization = nothing` discretizes `y` before the fit. Example:
  `ydiscr = Data.Gaussian.discretization_y()`
- `xdiscr::Function = nothing` with the signature `(d::DataFrame, t::DataFrame) ->
  (data[x]::Array, train[x]::Array, xlevels::Array)` discretizes `d` based on `t`, creating
  the `x` column in `data` and `train`. Example: `xdiscr = Unfold.run_treediscr(y, num_clusters)`
"""
function run{T1 <: Number, T2 <: Number}(data::DataFrame, train::DataFrame, y::Symbol, x::Symbol;
                                         method::Symbol=:expand, factor::Int64=1, keepdim::Bool=true,
                                         ydiscr::Union{Discretization, Void} = nothing,
                                         xdiscr::Union{Function, Void} = nothing,
                                         ylevels::AbstractArray{T1, 1} = ydiscr == nothing ? sort(unique(train[y])) : Float64[],
                                         xlevels::AbstractArray{T2, 1} = xdiscr == nothing ? sort(unique(train[x])) : Float64[],
                                         inspect::Union{Function, Void} = nothing,
                                         kwargs...)
    
    # discretize data with expanded levels
    if method == :expand && factor > 1
        if ydiscr == nothing || xdiscr == nothing
            error("RUN arguments 'ydiscr' and 'xdiscr' have to be set for method = :expand")
        elseif length(unique(train[y])) <= length(levels(ydiscr))
            warn("RUN input data may be discrete in $y - expansion will have no effect")
        end
        ydiscr = Discretization(ydiscr, num_bins = factor * length(levels(ydiscr)))
    elseif method != :reduce && factor > 1
        error("RUN method = :$method is not available")
    elseif factor < 1
        error("RUN factor has to be >= 1")
    end
    if ydiscr != nothing
        ylevels = levels(ydiscr)
        # info("Discretizing data for RUN with $(length(ylevels)) bins in $y")
        train = discretize(train, ydiscr)
    end
    if xdiscr != nothing
        # info("Discretizing RUN data in $x")
        data[x], train[x], xlevels = xdiscr(data,  train)
        # info("Obtained $(length(xlevels)) bins in $x")
    end
    
    # estimate transfer and spectrum of observation
    R = empiricaltransfer(train, y, x, ylevels = ylevels, xlevels = xlevels)
    g = histogram(data[x], xlevels)
    
    # advice from [blobel2002unfolding]: n_df is half the dimensionality of y.
    # After reducing the effective number of degrees of freedom to this value, combine
    # bins to obtain an estimate with half the number of initial bins. These are independent
    # from each other.
    fdim = size(R, 2)           # number of bins in f
    n_df = floor(fdim / factor) # desired number of degrees of freedom
    # info("Performing RUN on $fdim bins in $y with n_df = $(n_df)")
    if fdim % factor != 0 warn("dim(f) mod $factor should be zero!") end
    
    # post-processing for inspection and return value: expansion or reduction
    pp = f ->  if method == :expand && !keepdim
                   DataFrame(level = ylevels; (y, f))
               else # method == :reduce || keepdim
                   cdf = _combine(f, factor, ylevels, y)
                   if method == :expand || !keepdim
                       cdf
                   else # method == :reduce && keepdim
                       DataFrame(level = ylevels;
                                 (y, vcat([ repmat([i], factor) for i in cdf[y] ]...)[1:length(ylevels)]))
                   end
               end
    ppinspect = inspect
    if ppinspect != nothing
        ppinspect = (k, tau, ldiff, f) -> inspect(k, tau, ldiff, pp(f))
    end
    
    # estimate y
    return pp(run(R, g, n_df; inspect = ppinspect, kwargs...))
    
    
end

"""
    run(R, g, n_df = size(R, 2); kwargs...)

Perform RUN with the observed spectrum `g`, the detector response matrix `R`, and `n_df`
degrees of freedom. The default `n_df` results in no regularization; there is one degree of
freedom for each dimension in the result.

**Keyword arguments**

- `K = 100` specifies the maximum number of iterations
- `epsilon = 1e-6` specifies the minimum difference in the loss between iterations. RUN
  stops when the absolute loss difference drops below `epsilon`.
- `inspect` is an optional function `(k::Int, tau::Float, ldiff::Float, f_est::Array{Float, 1}) -> Any`
  called in every iteration `k`.
"""
function run{T<:Number}(R::AbstractArray{Float64,2}, g::AbstractArray{T,1},
                        n_df::Number = size(R, 2);
                        K::Int=100, epsilon::Float64=1e-6,
                        inspect::Union{Function, Void}=nothing)
    
    if any(g .<= 0) # limit unfolding to non-zero bins
        nonzero = g .> 0
        warn("Limiting RUN to $(sum(nonzero)) of $(length(g)) observeable non-zero bins")
        g = g[nonzero]
        R = R[nonzero, :]
    end
    if any([ all(R[j,:] .<= 0) for j in size(R, 1) ])
        warn("RUN response matrix has zero observeable bins")
    end
    if any([ all(R[:,i] .<= 0) for i in size(R, 2) ])
        warn("RUN response matrix has zero target bins")
    end
    fdim = size(R, 2) # dimension of f
    if fdim > size(R, 1)
        warn("RUN will go bad when performed on more target than observable bins")
    end
    
    # set up the loss function
    l   = _maxlikelihood_fct(R, g)  # the objective function,
    g_l = _maxlikelihood_grad(R, g) # ...its gradient,
    H_l = _maxlikelihood_hess(R, g) # ...and its Hessian.
    C   = _tikhonov_binning(fdim)   # the Tikhonov matrix (not in l and its derivatives)
    
    # initial estimate is zero vector
    f_est = zeros(Float64, fdim)
    if inspect != nothing
        inspect(0, NaN, NaN, f_est)
    end
    
    # the first iteration (least squares fit)
    H_lsq = _leastsquares_hess(R, g)(f_est)
    if any(isnan.(H_lsq)) || any(abs.(H_lsq) .== Inf)
        warn("LSq hessian contains Infs or NaNs - replacing these by zero")
        H_lsq[isnan.(H_lsq)] = 0.0
        H_lsq[abs.(H_lsq) .== Inf] = 0.0
    end
    f_est += try
        - inv(H_lsq) * _leastsquares_grad(R, g)(f_est)
    catch err
        if isa(err, Base.LinAlg.SingularException) # pinv instead of inv only required if more y than x bins
            warn("LSq hessian is singular - using pseudo inverse in RUN")
            - pinv(H_lsq) * _leastsquares_grad(R, g)(f_est)
        else
            rethrow(err)
        end
    end
    # f_est = pinv(R) * g # minimum-norm least squares solution
    if inspect != nothing
        inspect(1, NaN, NaN, f_est)
    end
    
    # subsequent iterations (maximum likelihood)
    l_1 = l(f_est)
    for k in 2:K
        
        g_f = g_l(f_est)
        H_f = H_l(f_est)
        if any(isnan.(H_f)) || any(abs.(H_f) .== Inf)
            warn("MaxL hessian contains Infs or NaNs - replacing these by zero")
            H_f[isnan.(H_f)] = 0.0
            H_f[abs.(H_f) .== Inf] = 0.0
        end
        
        # eigendecomposition of the Hessian: H_f == U*D*U' (complex conversion required if more y than x bins)
        eigvals_H, U = eig(H_f)
        D = diagm(real.(complex.(eigvals_H) .^ (-1/2))) # D^(-1/2)
        
        # eigendecomposition of transformed Tikhonov matrix: C2 == U_C*S*U_C'
        eigvals_C, U_C = eig(Symmetric( D*U' * C * U*D ))
        
        # select tau (special case: no regularization if n_df == fdim)
        tau = n_df < fdim ? _tau(n_df, eigvals_C) : 0.0
        
        # The step in the transformed problem and transformation to the actual solution.
        # The numeric difficulty with this approach is that the eigendecomposition
        # introduces error. Therefore, we only choose tau in the transformed problem and
        # take the step in the original problem instead of the commented-out solution.
        # 
        # S = diagm(eigvals_C)
        # f_2   = 1/2 * inv(eye(S) + tau*S) * (U*D*U_C)' * (H_f * f_est - g_f)
        # f_est = (U*D*U_C) * f_2
        f_est += try
            - inv(_maxlikelihood_hess(R, g, tau, C)(f_est)) * _maxlikelihood_grad(R, g, tau, C)(f_est)
        catch err
            if isa(err, Base.LinAlg.SingularException) # pinv instead of inv only required if more y than x bins
                warn("MaxL hessian is singular - using pseudo inverse in RUN")
                - pinv(_maxlikelihood_hess(R, g, tau, C)(f_est)) * _maxlikelihood_grad(R, g, tau, C)(f_est)
            else
                rethrow(err)
            end
        end
        
        # compute improvement in loss function and perform inspection
        l_f   = l(f_est)
        ldiff = l_1 - l_f
        l_1   = l_f
        if inspect != nothing
            inspect(k, tau, ldiff, f_est)
        end
        
         # stop when convergence criterion is met
        if abs(ldiff) < epsilon
            # info("Assuming RUN convergence in iteration $k from ldiff = $ldiff < epsilon = $epsilon")
            break
        elseif k == K
            # info("Stopping RUN after the maximum number K = $K of iterations")
        end
        
    end
    return f_est
    
end

# Brute-force search of a tau satisfying the n_df relation
function _tau(n_df::Number, eigvals_C::Array{Float64,1})
    res = _tau(n_df, tau -> sum([ 1/(1 + tau*v) for v in eigvals_C ])) # recursive subroutine
    # println("n_df = $(res[2]),  tau = $(res[1]) (",
    #         join(map(x -> "+-$x", res[3]), ", "), ")")
    return res[1] # final tau
end

# Recursive subroutine of _tau(::Number, ::Array{Float64,1}) for higher precision
function _tau(n_df::Number, taufunction::Function, min::Float64=-.01, max::Float64=-18.0, i::Int64=2)
    taus = logspace(min, max, 1000)
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

# Combine neighboring bins into one bin. n gives the number of neighbors that are combined.
# If levels and target are provided, return a histogram DataFrame as result.
function _combine{T <: Number}(y_vec::AbstractArray{T,1}, n::Int64,
                               ylevels::AbstractArray{Float64,1}, y::Symbol)
    maxindex  = length(ylevels) - n + 1
    indices   = 1:n:maxindex
    remainder = length(ylevels) % n
    if remainder > 0
        indices = vcat(indices, maxindex + 1)
    end
    return DataFrame(level = ylevels[indices];
                     (y, map(i -> sum(y_vec[i:min(length(y_vec), i + n - 1)]), indices))) # combined bins
end

"""
    run_treediscr(y, J)

Return a function `xdiscr` that can be used as an argument to `run`. The returned function
discretizes the observables of a data set with a decision tree trained to predict `y`, which
has `J` leaves (also see `TreeDiscretization`).
"""
run_treediscr(y::Symbol, J::Int) = (data, train) -> begin
    discr = TreeDiscretization(train, y, J)
    (discretize(data, discr), discretize(train, discr), levels(discr))
end # TODO move to ml.jl


"""
    alphadecay_exp(eta::Float64, start::Float64=1.0)

Return a `Function` object with the signature required by the `alpha` parameter in
`Unfold.dsea()`. Will reduce the `start` stepsize taken in iteration 1 by `eta` in each
subsequent iteration, i.e.,

    alpha = start * eta^(k-1).
"""
alphadecay_exp(eta::Float64, start::Float64=1.0) =
    (k::Int, pk::AbstractArray{Float64,1}, lastspec::AbstractArray{Float64,1}) -> start * eta^(k-1)

"""
    alphadecay_mul(eta::Float64, start::Float64=1.0)

Return a `Function` object with the signature required by the `alpha` parameter in
`Unfold.dsea()`. Computes alpha as follows:

    alpha = start * k ^ (eta-1)
    
For example, eta=.5 yields alpha = 1/sqrt(k). The exponent is chosen so that different
values of eta have a similar impact on alpha as in `alphadecay_exp`, i.e., values close
to 1 mean slow decay, values close to 0 mean fast decay.
"""
alphadecay_mul(eta::Float64, start::Float64=1.0) =
    (k::Int, pk::AbstractArray{Float64,1}, lastspec::AbstractArray{Float64,1}) -> start * k^(eta-1)



# objective function: negative log-likelihood
_maxlikelihood_fct{T <: Number}(R::Array{Float64,2}, g::AbstractArray{T,1},
                                tau::Float64=0.0,
                                C::Array{Float64,2}=zeros(size(R, 2), size(R, 2))) =
    f -> sum(begin
        fj = dot(R[j,:], f)
        fj - g[j]*real(log(complex(fj)))
    end for j in 1:length(g)) + tau/2 * dot(f, C*f)

# gradient of objective
_maxlikelihood_grad{T <: Number}(R::Array{Float64,2}, g::AbstractArray{T,1},
                                 tau::Float64=0.0,
                                 C::Array{Float64,2}=zeros(size(R, 2), size(R, 2))) =
    f -> [ sum([ R[j,i] - g[j]*R[j,i] / dot(R[j,:], f) for j in 1:length(g) ])
           for i in 1:length(f) ] + tau * C * f

# hessian of objective
_maxlikelihood_hess{T <: Number}(R::Array{Float64,2}, g::AbstractArray{T,1},
                                 tau::Float64=0.0,
                                 C::Array{Float64,2}=zeros(size(R,2), size(R,2))) =
    f -> begin
        res = zeros(Float64, (length(f), length(f)))
        for i1 in 1:length(f), i2 in 1:length(f)
            res[i1,i2] = sum( # hessian in cell (i1,i2)
                g[j]*R[j,i1]*R[j,i2] / dot(R[j,:], f)^2
            for j in 1:length(g))
        end
        full(Symmetric(res + tau * C))
    end



# objective function: least squares
_leastsquares_fct{T <: Number}(R::Array{Float64,2}, g::AbstractArray{T,1},
                               tau::Float64=0.0,
                               C::Array{Float64,2}=zeros(size(R, 2), size(R, 2))) =
    f -> sum([ (g[j] - dot(R[j,:], f))^2 / g[j] for j in 1:length(g) ])/2 + tau/2 * dot(f, C*f)

# gradient of objective
_leastsquares_grad{T <: Number}(R::Array{Float64,2}, g::AbstractArray{T,1},
                                tau::Float64=0.0,
                                C::Array{Float64,2}=zeros(size(R, 2), size(R, 2))) =
    f -> [ sum([ -R[j,i] * (g[j] - dot(R[j,:], f)) / g[j] for j in 1:length(g) ])
           for i in 1:length(f) ] + tau * C * f

# hessian of objective
_leastsquares_hess{T <: Number}(R::Array{Float64,2}, g::AbstractArray{T,1},
                                tau::Float64=0.0,
                                C::Array{Float64,2}=zeros(size(R, 2), size(R, 2))) =
    f -> begin
        res = zeros(Float64, (length(f), length(f)))
        for i1 in 1:length(f), i2 in 1:length(f)
            res[i1,i2] = sum( # hessian in cell (i1,i2)
                R[j,i1]*R[j,i2] / g[j]
            for j in 1:length(g))
        end
        full(Symmetric(res + tau * C))
    end

# Construct a Tikhonov matrix for binned discretization, as given in [cowan1998statistical, p. 169].
# This is equivalent to the notation in [blobel2002unfolding_long]!
_tikhonov_binning(matsize::Int)   = convert(Array{Float64,2},
        diagm(vcat([1, 5], repeat([6], inner=matsize-4), [5, 1])) +
        diagm(vcat([-2], repeat([-4], inner=matsize-3), [-2]),  1) +
        diagm(vcat([-2], repeat([-4], inner=matsize-3), [-2]), -1) +
        diagm(repeat([1], inner=matsize-2),  2) +
        diagm(repeat([1], inner=matsize-2), -2))

