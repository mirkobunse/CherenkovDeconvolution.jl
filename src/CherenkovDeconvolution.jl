# 
# CherenkovDeconvolution.jl
# Copyright 2018 Mirko Bunse
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
module CherenkovDeconvolution


using LinearAlgebra, DataFrames
using Optim: optimize # Optim required for adaptive step sizes

export Util, Sklearn
export dsea, ibu, run
export alpha_decay_exp, alpha_decay_mul, alpha_adaptive_run


# utility modules
include("util.jl")
include("sklearn.jl")


# deconvolution methods
include("methods/run.jl")
include("methods/ibu.jl")
include("methods/dsea.jl")


# 
# additional helpers
# 

# check and repair the f_0 argument
function _check_prior(f_0::Array{Float64,1}, m::Int64)
    if length(f_0) == 0
        return ones(m) ./ m
    elseif length(f_0) != m
        throw(DimensionMismatch("dim(f_0) = $(length(f_0)) != $m, the number of classes"))
    else # f_0 is provided and alright
        return Util.normalizepdf(f_0) # ensure pdf
    end
end

_check_prior(f_0::Array{Float64,1}, recode_dict::Dict) =
    _check_prior(length(f_0) > 0 ? f_0[sort(setdiff(collect(values(recode_dict)), [-1]))] : f_0, length(recode_dict)-1 )

# recode indices to resemble a unit range (no missing labels in between)
function _recode_indices(bins::AbstractArray{T,1}, inds::AbstractArray{T,1}...) where T<:Int
    
    # recode the training set
    inds_bins = sort(unique(vcat(inds...)))
    inds_dict = Dict(zip(inds_bins, 1:length(inds_bins)))
    inds_rec  = map(ind -> map(i -> inds_dict[i], ind), inds)
    
    # set up reverse recoding applied in _recode_result
    recode_dict = Dict(zip(values(inds_dict), keys(inds_dict))) # map from values to keys
    recode_dict[-1] = maximum(bins) # the highest actual bin (may not be in y_train)
    
    return recode_dict, inds_rec...
    
end

# recode a deconvolution result by reverting the initial recoding of the data
function _recode_result(f::Array{Float64,1}, recode_dict::Dict{T,T}) where T<:Int
    r = zeros(Float64, maximum(values(recode_dict)))
    for (k, v) in recode_dict
        if k != -1
            r[v] = f[k]
        end # else, the key was just included to store the maximum value
    end
    return r
end


end

