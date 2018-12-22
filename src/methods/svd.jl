"""
    svd(data, train, x, y[, bins]; kwargs...)

    svd(x_data, x_train, y_train[, bins]; kwargs...)

    svd(R, g; kwargs...)


Deconvolve the observed data applying the *SVD-based deconvolution algorithm* trained on the
given training set.

The first form of this function works on the two DataFrames `data` and `train`, where `y`
specifies the target column to be deconvolved (this column has to be present in `train`)
and `x` specifies the observed column present in both DataFrames. The second form accordingly
works on vectors and the third form makes use of a pre-defined detector response matrix `R`
and an observed (discrete) probability density `g`. In the first two forms, `R` and `g` are
directly obtained from the data and the keyword arguments.

The vectors `x_data`, `x_train`, and `y_train` (or accordingly `data[x]`, `train[x]`, and
`train[y]`) must contain label/observation indices rather than actual values. All expected
indices in `y_train` are optionally provided as `bins`.


**Keyword arguments**

- `effective_rank = -1`
  is a regularization parameter which defines the effective rank of the solution. This rank
  must be <= dim(f). Any value smaller than one results turns off regularization.
- `B = diagm(g)`
  is the co-variance matrix of the observed bins. The default value represents the
  assumption that each observed bin is Poisson-distributed with rate `g[i]`.
- `assume_poisson = true`
  is a short-hand for setting the covariance matrix `B` of the observed bins to `diagm(g)`
  (the default value) or to the unit matrix `eye(length(g)`. Using a unit matrix means to
  drop the Poisson assumption. If `B` is set explicitly, the value of `assume_poisson` is
  ignored.
- `epsilon_C = 1e-3`
  is a small constant to be added to each diagonal entry of the regularization matrix `C`.
  If no such constant would be added, inversion of `C` would not be possible.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.


**Caution:** According to the value of `fit_ratios`, the keyword argument `f_0` specifies a
ratio prior or a pdf prior, but only in the third form. In the second form, `f_0` always
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
svd( x_data  :: AbstractVector{T},
     x_train :: AbstractVector{T},
     y_train :: AbstractVector{T},
     bins_y  :: AbstractVector{T} = 1:maximum(y_train);
     kwargs... ) where T<:Int =
  _discrete_deconvolution(svd, x_data, x_train, y_train, bins_y, Dict{Symbol, Any}(kwargs))


function svd(R::Matrix{TR}, g::Vector{Tg};
             effective_rank::Int = -1,
             assume_poisson::Bool = true,
             B::Matrix{TB} = assume_poisson ? _svd_B_Poisson(g) : eye(eltype(g), length(g)),
             epsilon_C::Float64 = 1e-3,
             fit_ratios::Bool = false,
             kwargs...) where {TR<:Number, Tg<:Number, TB<:Number}
    
    # check arguments
    if size(R, 1) != length(g)
        throw(DimensionMismatch("dim(g) = $(length(g)) is not equal to the observable dimension $(size(R, 1)) of R"))
    elseif size(B, 1) != length(g) || size(B, 2) != length(g)
        throw(DimensionMismatch("One dimension dim(B) = $(size(B)) is not equal to the observable dimension $(length(g))"))
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

# co-variance matrix of the observed frequency distribution - assuming un-correlated Poisson-distributed bins
_svd_B_Poisson(g::Vector{T}) where T<:Number = diagm(g) # variance = mean, due to Poisson

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

