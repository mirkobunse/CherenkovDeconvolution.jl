import numpy as np

# initialize the Main namespace of the Julia interpreter
from pkg_resources import resource_filename
try:
    from julia import Main
except Exception as e:
    from julia.api import Julia # retry with compiled_modules=False
    jl = Julia(compiled_modules=False)
    from julia import Main
Main.eval(f'import Pkg; Pkg.activate("{resource_filename(__name__, "")}")')
Main.eval('using CherenkovDeconvolution')

# CherenkovDeconvolution.Methods
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

# CherenkovDeconvolution.Binnings
def TreeBinning(num_clusters, **kwargs):
    return Main.TreeBinning(num_clusters, **kwargs)
def KMeansBinning(num_clusters, **kwargs):
    return Main.KMeansBinning(num_clusters, **kwargs)

# CherenkovDeconvolution.OptimizedStepsizes and CherenkovDeconvolution.Stepsizes
def RunStepsize(binning, **kwargs):
    return Main.RunStepsize(binning, **kwargs)
def LsqStepsize(binning, **kwargs):
    return Main.LsqStepsize(binning, **kwargs)
def ConstantStepsize(alpha):
    return Main.ConstantStepsize(alpha)
def MulDecayStepsize(eta, alpha_0):
    return Main.MulDecayStepsize(eta, alpha_0)
def ExpDecayStepsize(eta, alpha_0):
    return Main.ExpDecayStepsize(eta, alpha_0)
