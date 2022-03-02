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
module Smoothings

using ..DeconvUtil, Polynomials

export apply, NoSmoothing, PolynomialSmoothing, Smoothing

"""
    abstract type Smoothing

Supertype of all smoothing strategies for intermediate priors.
"""
abstract type Smoothing end

"""
    apply(smoothing, f_est)

Apply the `smoothing` to the intermediate prior `f_est`.
"""
apply(s::Smoothing, f_est::AbstractVector{R}) where {R<:Real} =
    throw(ArgumentError("Implementation missing for $(typeof(s))")) # must be implemented for sub-types

"""
    NoSmoothing()

No smoothing; return the intermediate prior as it is.
"""
struct NoSmoothing <: Smoothing end

apply(s::NoSmoothing, f_est::AbstractVector{R}) where {R<:Real} = f_est

"""
    PolynomialSmoothing(order)

Intermediate priors are smoothed with a polynomial of the given `order`.

- `impact = 1.0` linearly interpolate between the smoothed and the actual
  prior if `0 < impact < 1` (default: use smoothed version).
- `avg_negative = true` replace negative values with the average of
  neighboring bins, as proposed in [dagostini2010improved]
- `warn = true` specifies if a warnings about negative values are emitted
"""
struct PolynomialSmoothing <: Smoothing
    order::Int
    impact::Float64
    warn::Bool
    avg_negative::Bool
    PolynomialSmoothing(
            order::Int;
            impact::Float64=1.0,
            warn=true,
            avg_negative=true) =
        new(order, impact, warn, avg_negative)
end

function apply(s::PolynomialSmoothing, f_est::AbstractVector{R}) where {R<:Real}
    if s.order < length(f_est)
        f = Polynomials.fit(Float64.(1:length(f_est)), f_est, s.order).(1:length(f_est))
        if s.avg_negative && any(f .< 0) # average values of neighbors for all values < 0
            for i in findall(f .< 0)
                f[i] = if i == 1
                           f[i+1] / 2
                       elseif i == length(f)
                           f[i-1] / 2
                       else
                           (f[i+1] + f[i-1]) / 4 # half the average [dagostini2010improved]
                       end
            end
            if s.warn # warn about negative values?
                Base.@warn "Values of neighbours averaged to circumvent negative values"
            end
        end
        f = DeconvUtil.normalizepdf(f, warn=s.warn)
        return s.impact * f + (1-s.impact) * f_est # linear interpolation
    else
        throw(ArgumentError("Impossible smoothing order $(s.order) >= dim(f_est) = $(length(f_est))"))
    end
end

end # module
