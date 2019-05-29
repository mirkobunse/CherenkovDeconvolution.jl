"""
    svd(data, train, x, y[, bins_y]; kwargs...)

    svd(x_data, x_train, y_train[, bins_y]; kwargs...)

    svd(R, g; kwargs...)


Deconvolve the observed data applying the *SVD-based deconvolution algorithm* trained on the
given training set.

The vectors `x_data`, `x_train`, and `y_train` (or accordingly `data[x]`, `train[x]`, and
`train[y]`) must contain label/observation indices rather than actual values. All expected
indices in `y_train` are optionally provided as `bins_y`. Alternatively, the detector
response matrix `R` and the observed density vector `g` can be given directly.


**Keyword arguments**

- `effective_rank = -1`
  is a regularization parameter which defines the effective rank of the solution. This rank
  must be <= dim(f). Any value smaller than one results turns off regularization.
- `N = length(x_data)`
  is the number of observations. In the third form of the method, `N=sum(g)` is the default,
  assuming that `g` contains absolute counts, not probabilities.
- `B = Util.cov_Poisson(g, N)`
  is the varianca-covariance matrix of the observed bins. The default value represents the
  assumption that each observed bin is Poisson-distributed with rate `g[i]*N`.
- `epsilon_C = 1e-3`
  is a small constant to be added to each diagonal entry of the regularization matrix `C`.
  If no such constant would be added, inversion of `C` would not be possible.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.


**Caution:** According to the value of `fit_ratios`, the keyword argument `f_0` specifies a
ratio prior or a pdf prior, but only in the third form. In the other forms, `f_0` always
specifies a pdf prior.
"""
svd( data   :: AbstractDataFrame,
     train  :: AbstractDataFrame,
     x      :: Symbol,
     y      :: Symbol,
     bins_y :: AbstractVector = 1:maximum(train[y]);
     kwargs... ) =
  svd(data[x], train[x], train[y], bins_y; kwargs...) # DataFrame form


# Vector form
function svd( x_data  :: AbstractVector{T},
              x_train :: AbstractVector{T},
              y_train :: AbstractVector{T},
              bins_y  :: AbstractVector{T} = 1:maximum(y_train);
              kwargs... ) where T<:Int
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :N) # number of observations must be set
        kwdict[:N] = length(x_data)
    end
    return _discrete_deconvolution(svd, x_data, x_train, y_train, bins_y, kwdict)
end

function svd(R::Matrix{TR}, g::Vector{Tg};
             effective_rank::Int = -1,
             N::Int = sum(Int64, g),
             B::Matrix{TB} = Util.cov_Poisson(g, N),
             epsilon_C::Float64 = 1e-3,
             fit_ratios::Bool = false,
             kwargs...) where {TR<:Number, Tg<:Number, TB<:Number}

    # check arguments
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    elseif size(B, 1) != length(g) || size(B, 2) != length(g)
        throw(DimensionMismatch("One of dim(B) = $(size(B)) is not equal to the observable dimension $(length(g))"))
    elseif effective_rank > size(R, 2)
        warn("Assuming effective_rank = $(size(R, 2)) instead of $(effective_rank)",
             " because effective_rank <= dim(f) is required")
        effective_rank = size(R, 2)
    end
    inv_C = inv(_svd_C(size(R, 2), epsilon_C))
    
    # 
    # Re-scaling and rotation steps 1-5 (without step 3) [hoecker1995svd]
    # 
    r, Q = eig(B) # transformation step 1
    R_tilde = diagm(sqrt.(r)) * Q' * R # sqrt by def (33), where R_ii = r_i^2
    g_tilde = diagm(sqrt.(r)) * Q' * g
    U, s, V = Base.svd(R_tilde * inv_C) # transformation step 4
    d = U' * g_tilde # transformation step 5
    
    # 
    # Deconvolution steps 2 and 3 [hoecker1995svd]
    # 
    # Step 1 is omitted because the effective_rank is already given, here. Step 4 is dealt
    # with in the _discrete_deconvolution wrapper.
    tau = (effective_rank > 0) ? s[effective_rank]^2 : 0.0 # deconvolution step 2
    z_tau = d .* s ./ ( s.^2 .+ tau )
    return inv_C * V * z_tau # step 3 (denoted as w_tau in the paper)
    
end

# regularization matrix C from the SVD approach - the square of _svd_C is similar but not
# equal to the tikhonov matrix from RUN
_svd_C(m::Int, epsilon::Float64) =
    (if m < 1
        throw(ArgumentError("m has to be greater than zero"))
    elseif m < 2 # stupid case
        eye(m)
    elseif m == 2 # not quite intelligent case
        -eye(m) + diagm(repeat([1], inner=m-1), 1) + diagm(repeat([1], inner=m-1), -1)
    else # usual case
        convert(Matrix{Float64},
            diagm(vcat([-1], repeat([-2], inner=m-2), [-1])) +
            diagm(repeat([1], inner=m-1),  1) +
            diagm(repeat([1], inner=m-1), -1))
    end) + epsilon * eye(m)

