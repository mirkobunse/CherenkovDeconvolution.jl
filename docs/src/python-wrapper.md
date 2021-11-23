# Python wrapper

The wrapper provides an interface through which the methods of CherenkovDeconvolution.jl are exposed to Python. This short guide assumes you have read the [Manual](@ref) section.


## Getting started

You need a working Julia installation on your system. To install Julia, please refer to [the official instructions](https://julialang.org/downloads/platform/#linux_and_freebsd). The wrapper, including all dependies, can then be installed with pip:

```
pip install https://github.com/mirkobunse/CherenkovDeconvolution.jl/archive/py.tar.gz
```

Python translations of the examples from the [Manual](@ref) section would be:

```python
import CherenkovDeconvolution_jl as cd
from sklearn.naive_bayes import GaussianNB

# DSEA+ example
dsea = cd.DSEA(GaussianNB())
f_dsea = cd.deconvolve(dsea, X_obs, X_trn, y_trn)

# classical deconvolution example
run = cd.RUN(cd.TreeBinning(10)) # use up to 10 clusters
f_run = cd.deconvolve(run, X_obs, X_trn, y_trn)

# adaptive stepsizes in DSEA+
stepsize = cd.RunStepsize(cd.TreeBinning(10); decay=true)
dsea_plus = cd.DSEA(GaussianNB(); K=100, epsilon=1e-6, stepsize=stepsize)
f_dsea_plus = cd.deconvolve(dsea_plus, X_obs, X_trn, y_trn)
```
