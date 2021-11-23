import numpy as np

# initialize the Main namespace of the Julia interpreter
from pkg_resources import resource_filename
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval(f'import Pkg; Pkg.activate("{resource_filename(__name__, "")}")')
Main.eval('import CherenkovDeconvolution.Methods: deconvolve, DSEA, IBU, PRUN, RUN, SVD')

def deconvolve(m, X_obs, X_trn, y_trn):
    return Main.deconvolve(m, X_obs, X_trn, np.array(y_trn) + 1)
def DSEA(classifier, **kwargs):
    return Main.DSEA(classifier, **kwargs)
def IBU(binning, **kwargs):
    return Main.IBU(binning, **kwargs)
def PRUN(binning, **kwargs):
    return Main.RUN(binning, **kwargs)
def RUN(binning, **kwargs):
    return Main.RUN(binning, **kwargs)
def SVD(binning, **kwargs):
    return Main.SVD(binning, **kwargs)
