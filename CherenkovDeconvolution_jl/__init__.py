import numpy as np
from importlib import reload
from pkg_resources import resource_filename

def __jl():
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    return Main

def __activate():
    project = resource_filename(__name__, "")
    __jl().eval(f'import Pkg; Pkg.activate("{project}")')

def install(version="main"):
    import julia
    julia.install()
    reload(julia) # reload required between julia.install and Main.eval
    __activate()
    __jl().eval(f'Pkg.add(url="https://github.com/mirkobunse/CherenkovDeconvolution.jl.git", rev="{version}")')

def initialize():
    __activate()
    __jl().eval('import CherenkovDeconvolution.Methods: deconvolve, DSEA, IBU, PRUN, RUN, SVD')

def deconvolve(m, X_obs, X_trn, y_trn):
    return __jl().deconvolve(m, X_obs, X_trn, np.array(y_trn) + 1)
def DSEA(classifier, **kwargs):
    return __jl().DSEA(classifier, **kwargs)
def IBU(binning, **kwargs):
    return __jl().IBU(binning, **kwargs)
def PRUN(binning, **kwargs):
    return __jl().RUN(binning, **kwargs)
def RUN(binning, **kwargs):
    return __jl().RUN(binning, **kwargs)
def SVD(binning, **kwargs):
    return __jl().SVD(binning, **kwargs)
