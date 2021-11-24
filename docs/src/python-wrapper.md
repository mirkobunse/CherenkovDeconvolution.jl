# Python wrapper

The wrapper provides an interface through which the methods of CherenkovDeconvolution.jl are exposed to Python. This short guide assumes you have read the [Manual](@ref) section.


## Installation

- **Step 1:** You need a working Julia installation on your system. To install Julia, please refer to [the official instructions](https://julialang.org/downloads/platform/#linux_and_freebsd).
- **Step 2:** The wrapper, including all dependencies, can then be installed via pip:

```
pip install https://github.com/mirkobunse/CherenkovDeconvolution.jl/archive/py.tar.gz
```


## Usage

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


## How it works

This wrapper is realized with [PyJulia](https://github.com/JuliaPy/pyjulia).

The pip installer takes care of preparing the existing Julia installation for the wrapper: it installs the Python bridge [PyCall.jl](https://github.com/JuliaPy/PyCall.jl), the wrapped package [CherenkovDeconvolution.jl](https://github.com/mirkobunse/CherenkovDeconvolution.jl), and all of their Julia package dependencies.

Calling `import CherenkovDeconvolution_jl` in Python starts a Julia process in the background, which executes all Julia methods with the argument values provided by Python and returns the results.


## Troubleshooting

The [PyJulia](https://github.com/JuliaPy/pyjulia) bridge documents [solutions to some of its frequent issues](https://pyjulia.readthedocs.io/en/stable/troubleshooting.html). Additional bug fixes are discussed in [the issues of PyJulia](https://github.com/JuliaPy/pyjulia/issues/).

Some of these solutions are already part of the installation process, see `setup.py` and `CherenkovDeconvolution_jl/init.py` in the `py` branch of this repository. If you encounter any other problems, please [file an issue at our package](https://github.com/mirkobunse/CherenkovDeconvolution.jl/issues/new).
