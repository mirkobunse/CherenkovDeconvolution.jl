# load data
using MLDataUtils
X, y_labels, _ = load_iris()

# discretize target quantity (use LinearDiscretizer for numerical values)
using Discretizers: encode, CategoricalDiscretizer
y = encode(CategoricalDiscretizer(y_labels), y_labels)

# split into training and observed set (obsdim is required because MLDataUtils uses transposed data)
(X_train, y_train), (X_data, y_data) = splitobs(shuffleobs((X', y), obsdim = 1), obsdim = 1)

# deconvolve
using ScikitLearn, CherenkovDeconvolution
@sk_import naive_bayes : GaussianNB
foo   = CherenkovDeconvolution.Sklearn.train_and_predict_proba(GaussianNB())
f_est = CherenkovDeconvolution.dsea(X_data, X_train, y_train, foo)



# Histograms
using StatsBase

data = rand(100) .* (maxval - minval) .+ minval # random numbers
pdf = normalize(fit(Histogram, data, binedges(lindisc), closed = :left), mode = :probability).weights # never forget mode!
