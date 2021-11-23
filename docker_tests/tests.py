import CherenkovDeconvolution_jl as cd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# generate random data
X_obs = np.random.rand(100, 5)
X_trn = np.random.rand(500, 5)
y_trn = np.random.randint(3, size=500)

# deconvolve
pcc = cd.DSEA(GaussianNB())
f_pcc = cd.deconvolve(pcc, X_obs, X_trn, y_trn)
print(f"The random deconvolution result of PCC is {f_pcc}")

dsea_plus = cd.DSEA(
    GaussianNB(),
    stepsize = cd.RunStepsize(cd.TreeBinning(10)),
    K = 10,
    epsilon = 1e-6
)
f_dsea_plus = cd.deconvolve(dsea_plus, X_obs, X_trn, y_trn)
print(f"The random deconvolution result of DSEA+ is {f_dsea_plus}")

ibu = cd.IBU(cd.TreeBinning(10))
f_ibu = cd.deconvolve(ibu, X_obs, X_trn, y_trn)
print(f"The random deconvolution result of IBU is {f_ibu}")

run = cd.RUN(cd.KMeansBinning(10))
f_run = cd.deconvolve(run, X_obs, X_trn, y_trn)
print(f"The random deconvolution result of RUN is {f_run}")
