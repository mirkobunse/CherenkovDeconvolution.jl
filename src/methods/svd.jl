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
export SVD

"""
    SVD(binning; kwargs...)

The *SVD-based* deconvolution method, using a `binning` to discretize the observable features.

**Keyword arguments**

- `effective_rank = -1`
  is a regularization parameter which defines the effective rank of the solution. This rank
  must be <= dim(f). Any value smaller than one results turns off regularization.
- `N = sum(g)`
  is the number of observations.
- `B = DeconvUtil.cov_Poisson(g, N)`
  is the varianca-covariance matrix of the observed bins. The default value represents the
  assumption that each observed bin is Poisson-distributed with rate `g[i]*N`.
- `epsilon_C = 1e-3`
  is a small constant to be added to each diagonal entry of the regularization matrix `C`.
  If no such constant would be added, inversion of `C` would not be possible.
- `fit_ratios = false`
  determines if ratios are fitted (i.e. `R` has to contain counts so that the ratio
  `f_est / f_train` is estimated) or if the probability density `f_est` is fitted directly.
"""
struct SVD <: DiscreteMethod
    binning :: Binning
    B :: Matrix{Float64}
    effective_rank :: Int
    epsilon_C :: Float64
    fit_ratios :: Bool
    n_bins_y :: Int
    N :: Int
    SVD(binning :: Binning;
        B          :: Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
        effective_rank :: Int = -1,
        epsilon_C  :: Float64 = 1e-3,
        fit_ratios :: Bool    = false,
        n_bins_y   :: Int     = -1,
        N          :: Int     = -1
    ) = new(binning, B, effective_rank, epsilon_C, fit_ratios, n_bins_y, N)
end

binning(svd::SVD) = svd.binning
expects_normalized_R(svd::SVD) = !svd.fit_ratios
expects_normalized_g(svd::SVD) = false
expected_n_bins_y(svd::SVD) = svd.n_bins_y

function deconvolve(
        svd::SVD,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f},
        f_0::Union{Vector{T_f},Nothing}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number}

    # check arguments
    check_discrete_arguments(R, g)
    N = svd.N > 0 ? svd.N : sum(g) # the configured value or the default
    B = length(svd.B) > 0 ? svd.B : DeconvUtil.cov_Poisson(g, N)
    effective_rank = svd.effective_rank
    if size(B, 1) != length(g) || size(B, 2) != length(g)
        throw(DimensionMismatch("One of dim(B) = $(size(B)) is not equal to the observable dimension $(length(g))"))
    elseif effective_rank > size(R, 2)
        @warn "Assuming effective_rank = $(size(R, 2)) instead of $(effective_rank) because effective_rank <= dim(f) is required"
        effective_rank = size(R, 2)
    end
    inv_C = LinearAlgebra.inv(_svd_C(size(R, 2), svd.epsilon_C))

    # 
    # Re-scaling and rotation steps 1-5 (without step 3) [hoecker1995svd]
    # 
    r, Q = LinearAlgebra.eigen(B) # transformation step 1
    R_tilde = Matrix(Diagonal(sqrt.(r))) * Q' * R # sqrt by def (33), where R_ii = r_i^2
    g_tilde = Matrix(Diagonal(sqrt.(r))) * Q' * g
    U, s, V = LinearAlgebra.svd(R_tilde * inv_C) # transformation step 4
    d = U' * g_tilde # transformation step 5

    # 
    # Deconvolution steps 2 and 3 [hoecker1995svd]
    # 
    # Step 1 is omitted because the effective_rank is already given, here. Step 4 is dealt
    # with in the _discrete_deconvolution wrapper.
    tau = (effective_rank > 0) ? s[effective_rank]^2 : 0.0 # deconvolution step 2
    z_tau = d .* s ./ ( s.^2 .+ tau )
    f_est = inv_C * V * z_tau # step 3 (denoted as w_tau in the paper)
    return DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f_est))
end

# regularization matrix C from the SVD approach - the square of _svd_C is similar but not
# equal to the tikhonov matrix from RUN
_svd_C(m::Int, epsilon::Float64) =
    (if m < 1
        throw(ArgumentError("m has to be greater than zero"))
    elseif m < 2 # stupid case
        Matrix(1.0I, m, m) # identity matrix
    else # usual case
        convert(Matrix{Float64}, diagm(
            0 => vcat([-1], repeat([-2], inner=m-2), [-1]),
            1 => repeat([1], inner=m-1),
           -1 => repeat([1], inner=m-1)
        ))
    end) + Matrix(epsilon * I, m, m)

# deprecated syntax (the IdentityBinning is defined in src/methods/run.jl)
export svd
function svd(
        x_obs  :: AbstractVector{T},
        x_trn  :: AbstractVector{T},
        y_trn  :: AbstractVector{T},
        bins_y :: AbstractVector{T} = 1:maximum(y_trn);
        kwargs...
        ) where T<:Int
    Base.depwarn(join([
        "`svd(data, config)` is deprecated; ",
        "please call `deconvolve(SVD(config), data)` instead"
    ]), :svd)
    svd = SVD(IdentityBinning(); n_bins_y=length(bins_y), kwargs...)
    return deconvolve(
        svd,
        reshape(x_obs, (length(x_obs), 1)), # treat as a matrix
        reshape(x_trn, (length(x_trn), 1)),
        y_trn
    )
end
