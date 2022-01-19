# API reference

The following is a list of all public methods in CherenkovDeconvolution.jl.

```@meta
CurrentModule = CherenkovDeconvolution
```

## Deconvolution methods

All deconvolution methods implement the [`deconvolve`](@ref) function.

```@docs
deconvolve
prefit
DSEA
IBU
PRUN
RUN
SVD
```

## Binnings

Binnings are needed by the classical (discrete) deconvolution algorithms, e.g. [`IBU`](@ref), [`PRUN`](@ref), [`RUN`](@ref), and [`SVD`](@ref).

```@docs
TreeBinning
KMeansBinning
ClassificationPreprocessor
DefaultPreprocessor
```

## Stepsizes

Stepsizes can be used in [`DSEA`](@ref) and [`IBU`](@ref). Combining the [`RunStepsize`](@ref) with [`DSEA`](@ref) yields the *DSEA+* version of the algorithm. More information on stepsizes is given in the [Manual](@ref).

```@docs
RunStepsize
LsqStepsize
ConstantStepsize
MulDecayStepsize
ExpDecayStepsize
DEFAULT_STEPSIZE
```

## DeconvUtil

The module `DeconvUtil` provides a rich set of user-level ulitity functions. We do not export the members of this module directly, so that you need to name the module when using its functions.

```julia
using CherenkovDeconvolution
fit_pdf([.3, .4, .3]) # WILL BREAK

# solution a)
DeconvUtil.fit_pdf([.3, .4, .3])

# solution b)
import DeconvUtil: fit_pdf
fit_pdf([.3, .4, .3])
```

```@docs
DeconvUtil.fit_pdf
DeconvUtil.fit_R
DeconvUtil.normalizetransfer
DeconvUtil.normalizepdf
DeconvUtil.normalizepdf!
DeconvUtil.polynomial_smoothing
```

## Developer interface

The following list of methods is primarily intended for developers who wish to implement their own deconvolution methods, binnings, stepsizes, etc. If you do so, please file a pull request so that others can benefit from your work! More information on how to develop for this package is given in the [Developer manual](@ref).

```@docs
DeconvolutionMethod
DiscreteMethod

Binning
BinningDiscretizer
bins
encode

Stepsize
OptimizedStepsize
initialize_prefit!
initialize_deconvolve!
value

check_prior
check_arguments
LoneClassException
recover_estimate

LabelSanitizer
encode_labels
encode_prior
decode_estimate
```
