[![Build Status](https://travis-ci.org/mirkobunse/CherenkovDeconvolution.jl.svg?branch=master)](https://travis-ci.org/mirkobunse/CherenkovDeconvolution.jl)

# CherenkovDeconvolution.jl

Deconvolution algorithms for Cherenkov astronomy and other use cases.



## Getting Started

This package is documented on [our deconvolution website](https://sfb876.tu-dortmund.de/deconvolution).
It is installed with the package manager (Julia-0.7 and above):

```julia
using Pkg
Pkg.clone("git://github.com/mirkobunse/CherenkovDeconvolution.jl.git")
```

The [example directory](https://github.com/mirkobunse/CherenkovDeconvolution.jl/tree/master/example)
explains, how to use the package.

**Caution:** Since [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) does not yet
support Julia-1.0, we can not provide full support either. However, apart from the
`CherenkovDeconvolution.Sklearn` sub-module, our package is ready for this major Julia release.



## Current Status

CherenkovDeconvolution.jl implements an enhanced version of the Dortmund Spectrum Estimation Algorithm (DSEA+),
the Regularized Unfolding (RUN) method, and the Iterative Bayesian Unfolding (IBU).
An extensive set of experiments is taken out on these algorithms [in another repository](https://github.com/mirkobunse/deconv-exp).

We also ported this package to Python, calling it [CherenkovDeconvolution.py](https://github.com/mirkobunse/CherenkovDeconvolution.py).

