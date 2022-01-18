# Developer manual

This section is meant to give an idea of how to implement new algorithms in the framework of this package. Examples of all the following descriptions can be found in the original source code.


## Deconvolution methods

You can create a new deconvolution method in two steps:

- create a subtype of [`DeconvolutionMethod`](@ref)
- implement the [`deconvolve`](@ref) function for this subtype

```julia
struct MyNewMethod <: DeconvolutionMethod
    ...
end

function CherenkovDeconvolution.deconvolve(
        m::MyNewMethod,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer}
    ...
end
```

To create a *discrete* deconvolution method, which solves `g = R * f`, you can alternatively create a subtype of [`DiscreteMethod`](@ref), for which the [`deconvolve`](@ref) function is allowed to look differently:

```julia
struct MyNewDiscreteMethod <: DiscreteMethod
    ...
end

function CherenkovDeconvolution.deconvolve(
        m::MyNewDiscreteMethod,
        R::Matrix{T_R},
        g::Vector{T_g},
        label_sanitizer::LabelSanitizer,
        f_trn::Vector{T_f},
        f_0::Union{Vector{T_f},Nothing}
        ) where {T_R<:Number,T_g<:Number,T_f<:Number}
    ...
end
```


## Binnings

[`Binning`](@ref) methods are created in a very similar fashion, but they require one extra step: a [`BinningDiscretizer`](@ref) must be derived from the binning and this discretizer must be used in an implementation of [`encode`](@ref).

```julia
struct MyNewBinning <: Binning
    ...
end

struct MyNewDiscretizer{T} <: BinningDiscretizer{T}
    ...
end

function CherenkovDeconvolution.BinningDiscretizer(
        b::MyNewBinning,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer}
    ...
    return MyNewDiscretizer{T}(...)
end

function Discretizers.encode(d::MyNewDiscretizer{T}, X_obs::Any)
    ...
end
```


## Stepsizes

[`Stepsize`](@ref) methods require an implementation of the [`value`](@ref) function. Optionally, they can also implement [`initialize_prefit!`](@ref) and [`initialize_deconvolve!`](@ref).

```julia
struct MyNewStepsize <: Stepsize
    ...
end

function CherenkovDeconvolution.value(
        s::MyNewStepsize,
        k::Int,
        p::Vector{Float64},
        f::Vector{Float64},
        a::Float64)
    ...
end

# optional
function CherenkovDeconvolution.initialize_prefit!(
        s::MyNewStepsize,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer}
    ...
end
function CherenkovDeconvolution.initialize_deconvolve!(
        s::MyNewStepsize,
        X_obs::Any,
        )
    ...
end
```
