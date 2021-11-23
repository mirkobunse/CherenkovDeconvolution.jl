import CherenkovDeconvolution_jl as cd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# generate random data
X_obs = np.random.rand(100, 5)
X_trn = np.random.rand(500, 5)
y_trn = np.random.randint(3, size=500)

# initialize CherenkovDeconvolution_jl
cd.install()
cd.initialize()

# deconvolve
dsea = cd.DSEA(GaussianNB())
f_est = cd.deconvolve(dsea, X_obs, X_trn, y_trn)
print(f"The random deconvolution result of DSEA is {f_est}")
