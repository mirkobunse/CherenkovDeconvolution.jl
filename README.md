[![Build Status](https://github.com/mirkobunse/CherenkovDeconvolution.jl/workflows/CI/badge.svg)](https://github.com/mirkobunse/CherenkovDeconvolution.jl/actions)

# CherenkovDeconvolution.jl

Deconvolution estimates the probability density function of a latent quantity
by relating this quantity to other, measurable quantities.
It is frequently used in Cherenkov astronomy, where the energy distribution
of cosmic particles is estimated from data taken by a telescope.


## Getting Started

You can install this package with the Julia package manager (Julia-0.7 and above):

```julia
] add CherenkovDeconvolution
```

The [docs/](https://github.com/mirkobunse/CherenkovDeconvolution.jl/tree/master/docs)
directory provides you with some usage examples.
More information can be found on [our website](https://sfb876.tu-dortmund.de/deconvolution).


## Current Status

CherenkovDeconvolution.jl implements our enhanced version of the Dortmund Spectrum Estimation Algorithm (DSEA+),
the Regularized Unfolding (RUN) method, and the Iterative Bayesian Unfolding (IBU).
An extensive set of experiments is taken out on these algorithms [in another repository](https://github.com/mirkobunse/deconv-exp).

We also ported this package to Python, calling it [CherenkovDeconvolution.py](https://github.com/mirkobunse/CherenkovDeconvolution.py).
