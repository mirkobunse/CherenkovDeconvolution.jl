var documenterSearchIndex = {"docs":
[{"location":"developer-manual/#Developer-manual","page":"Developer manual","title":"Developer manual","text":"","category":"section"},{"location":"api-reference/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"The following is a list of all public methods in CherenkovDeconvolution.jl.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"CurrentModule = CherenkovDeconvolution","category":"page"},{"location":"api-reference/#Deconvolution-methods","page":"API reference","title":"Deconvolution methods","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"All deconvolution methods implement the deconvolve function.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"deconvolve\nDSEA\nIBU\nPRUN\nRUN\nSVD","category":"page"},{"location":"api-reference/#CherenkovDeconvolution.Methods.deconvolve","page":"API reference","title":"CherenkovDeconvolution.Methods.deconvolve","text":"deconvolve(m, X_obs, X_trn, y_trn)\n\nDeconvolve the observed features in X_obs with the deconvolution method m trained on the features X_trn and the corresponding labels y_trn.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Methods.DSEA","page":"API reference","title":"CherenkovDeconvolution.Methods.DSEA","text":"DSEA(classifier; kwargs...)\n\nThe DSEA/DSEA+ deconvolution method, embedding the given classifier.\n\nKeyword arguments\n\nf_0 = ones(m) ./ m defines the prior, which is uniform by default\nfixweighting = true sets, whether or not the weight update fix is applied. This fix is proposed in my Master's thesis and in the corresponding paper.\nstepsize = DEFAULT_STEPSIZE is the step size taken in every iteration.\nsmoothing = Base.identity is a function that optionally applies smoothing in between iterations.\nK = 1 is the maximum number of iterations.\nepsilon = 0.0 is the minimum symmetric Chi Square distance between iterations. If the actual distance is below this threshold, convergence is assumed and the algorithm stops.\ninspect = nothing is a function (f_k::Vector, k::Int, chi2s::Float64, alpha_k::Float64) -> Any optionally called in every iteration.\nreturn_contributions = false sets, whether or not the contributions of individual examples in X_obs are returned as a tuple together with the deconvolution result.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Methods.IBU","page":"API reference","title":"CherenkovDeconvolution.Methods.IBU","text":"IBU(binning; kwargs...)\n\nThe Iterative Bayesian Unfolding deconvolution method, using a binning to discretize the observable features.\n\nKeyword arguments\n\nf_0 = ones(m) ./ m defines the prior, which is uniform by default.\nsmoothing = Base.identity is a function that optionally applies smoothing in between iterations. The operation is neither applied to the initial prior, nor to the final result. The function inspect is called before the smoothing is performed.\nK = 3 is the maximum number of iterations.\nepsilon = 0.0 is the minimum symmetric Chi Square distance between iterations. If the actual distance is below this threshold, convergence is assumed and the algorithm stops.\nstepsize = DEFAULT_STEPSIZE is the step size taken in every iteration.\ninspect = nothing is a function (f_k::Vector, k::Int, chi2s::Float64, alpha_k::Float64) -> Any optionally called in every iteration.\nfit_ratios = false (discouraged) determines if ratios are fitted (i.e. R has to contain counts so that the ratio f_est / f_train is estimated) or if the probability density f_est is fitted directly.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Methods.PRUN","page":"API reference","title":"CherenkovDeconvolution.Methods.PRUN","text":"PRUN(binning; kwargs...)\n\nA version of the Regularized Unfolding method that is constrained to positive results. Like the original version, it uses a binning to discretize the observable features.\n\nKeyword arguments\n\ntau = 0.0 determines the regularisation strength.\nK = 100 is the maximum number of iterations.\nepsilon = 1e-6 is the minimum difference in the loss function between iterations. RUN stops when the absolute loss difference drops below epsilon.\nf_0 = ones(size(R, 2)) Starting point for the interior-point Newton optimization.\nacceptance_correction = nothing  is a tuple of functions (ac(d), invac(d)) representing the acceptance correction ac and its inverse operation invac for a data set d.\nac_regularisation = true  decides whether acceptance correction is taken into account for regularisation. Requires acceptance_correction != nothing.\nlog_constant = 1/18394 is a selectable constant used in log regularisation to prevent the undefined case log(0).\ninspect = nothing is a function (f_k::Vector, k::Int, ldiff::Float64) -> Any called in each iteration.\nfit_ratios = false (discouraged) determines if ratios are fitted (i.e. R has to contain counts so that the ratio f_est / f_train is estimated) or if the probability density f_est is fitted directly.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Methods.RUN","page":"API reference","title":"CherenkovDeconvolution.Methods.RUN","text":"RUN(binning; kwargs...)\n\nThe Regularized Unfolding method, using a binning to discretize the observable features.\n\nKeyword arguments\n\nn_df = size(R, 2) is the effective number of degrees of freedom. The default n_df results in no regularization (there is one degree of freedom for each dimension in the result).\nK = 100 is the maximum number of iterations.\nepsilon = 1e-6 is the minimum difference in the loss function between iterations. RUN stops when the absolute loss difference drops below epsilon.\nacceptance_correction = nothing  is a tuple of functions (ac(d), invac(d)) representing the acceptance correction ac and its inverse operation invac for a data set d.\nac_regularisation = true  decides whether acceptance correction is taken into account for regularisation. Requires acceptance_correction != nothing.\nlog_constant = 1/18394 is a selectable constant used in log regularisation to prevent the undefined case log(0).\ninspect = nothing is a function (f_k::Vector, k::Int, ldiff::Float64, tau::Float64) -> Any optionally called in every iteration.\nfit_ratios = false (discouraged) determines if ratios are fitted (i.e. R has to contain counts so that the ratio f_est / f_train is estimated) or if the probability density f_est is fitted directly.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Methods.SVD","page":"API reference","title":"CherenkovDeconvolution.Methods.SVD","text":"SVD(binning; kwargs...)\n\nThe SVD-based deconvolution method, using a binning to discretize the observable features.\n\nKeyword arguments\n\neffective_rank = -1 is a regularization parameter which defines the effective rank of the solution. This rank must be <= dim(f). Any value smaller than one results turns off regularization.\nN = sum(g) is the number of observations.\nB = DeconvUtil.cov_Poisson(g, N) is the varianca-covariance matrix of the observed bins. The default value represents the assumption that each observed bin is Poisson-distributed with rate g[i]*N.\nepsilon_C = 1e-3 is a small constant to be added to each diagonal entry of the regularization matrix C. If no such constant would be added, inversion of C would not be possible.\nfit_ratios = true determines if ratios are fitted (i.e. R has to contain counts so that the ratio f_est / f_train is estimated) or if the probability density f_est is fitted directly.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#Binnings","page":"API reference","title":"Binnings","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"Binnings are needed by the classical (discrete) deconvolution algorithms, e.g. IBU, PRUN, RUN, and SVD.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"TreeBinning\nKMeansBinning","category":"page"},{"location":"api-reference/#CherenkovDeconvolution.Binnings.TreeBinning","page":"API reference","title":"CherenkovDeconvolution.Binnings.TreeBinning","text":"TreeBinning(J; kwargs...)\n\nA supervised tree binning strategy with up to J clusters.\n\nKeyword arguments\n\ncriterion = \"gini\" is the splitting criterion of the tree.\nseed = rand(UInt32) is the random seed for tie breaking.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Binnings.KMeansBinning","page":"API reference","title":"CherenkovDeconvolution.Binnings.KMeansBinning","text":"KMeansBinning(J; seed=rand(UInt32))\n\nAn unsupervised binning strategy with up to J clusters.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#Stepsizes","page":"API reference","title":"Stepsizes","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"Stepsizes can be used in DSEA and IBU. Combining the RunStepsize with DSEA yields the DSEA+ version of the algorithm. More information on stepsizes is given in the Manual.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"RunStepsize\nLsqStepsize\nConstantStepsize\nMulDecayStepsize\nExpDecayStepsize\nDEFAULT_STEPSIZE","category":"page"},{"location":"api-reference/#CherenkovDeconvolution.OptimizedStepsizes.RunStepsize","page":"API reference","title":"CherenkovDeconvolution.OptimizedStepsizes.RunStepsize","text":"RunStepsize(binning; kwargs...)\n\nAdapt the step size by maximizing the likelihood of the next estimate in the search direction of the current iteration, much like in the RUN deconvolution method.\n\nKeyword arguments:\n\ndecay = false specifies whether a_k+1 <= a_k is enforced so that step sizes never increase.\ntau = 0.0 determines the regularisation strength.\nwarn = false specifies whether warnings should be emitted for debugging purposes.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.OptimizedStepsizes.LsqStepsize","page":"API reference","title":"CherenkovDeconvolution.OptimizedStepsizes.LsqStepsize","text":"LsqStepsize(binning; kwargs...)\n\nAdapt the step size by solving a least squares objective in the search direction of the current iteration.\n\nKeyword arguments:\n\ndecay = false specifies whether a_k+1 <= a_k is enforced so that step sizes never increase.\ntau = 0.0 determines the regularisation strength.\nwarn = false specifies whether warnings should be emitted for debugging purposes.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Stepsizes.ConstantStepsize","page":"API reference","title":"CherenkovDeconvolution.Stepsizes.ConstantStepsize","text":"ConstantStepsize(alpha)\n\nChoose the constant step size alpha in every iteration.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Stepsizes.MulDecayStepsize","page":"API reference","title":"CherenkovDeconvolution.Stepsizes.MulDecayStepsize","text":"MulDecayStepsize(eta, a=1.0)\n\nReduce the first stepsize a by eta in each iteration:\n\nvalue(MulDecayStepsize(eta, a), k, ...) == a * k^(eta-1)\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Stepsizes.ExpDecayStepsize","page":"API reference","title":"CherenkovDeconvolution.Stepsizes.ExpDecayStepsize","text":"ExpDecayStepsize(eta, a=1.0)\n\nReduce the first stepsize a by eta in each iteration:\n\nvalue(ExpDecayStepsize(eta, a), k, ...) == a * eta^(k-1)\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Stepsizes.DEFAULT_STEPSIZE","page":"API reference","title":"CherenkovDeconvolution.Stepsizes.DEFAULT_STEPSIZE","text":"const DEFAULT_STEPSIZE = ConstantStepsize(1.0)\n\nThe default stepsize in all deconvolution methods.\n\n\n\n\n\n","category":"constant"},{"location":"api-reference/#DeconvUtil","page":"API reference","title":"DeconvUtil","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"The module DeconvUtil provides a rich set of user-level ulitity functions. We do not export the members of this module directly, so that you need to name the module when using its functions.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"using CherenkovDeconvolution\nfit_pdf([.3, .4, .3]) # WILL BREAK\n\n# solution a)\nDeconvUtil.fit_pdf([.3, .4, .3])\n\n# solution b)\nimport DeconvUtil: fit_pdf\nfit_pdf([.3, .4, .3])","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"DeconvUtil.fit_pdf\nDeconvUtil.fit_R\nDeconvUtil.normalizetransfer\nDeconvUtil.normalizepdf\nDeconvUtil.normalizepdf!\nDeconvUtil.polynomial_smoothing","category":"page"},{"location":"api-reference/#CherenkovDeconvolution.DeconvUtil.fit_pdf","page":"API reference","title":"CherenkovDeconvolution.DeconvUtil.fit_pdf","text":"fit_pdf(x[, bins]; normalize=true, laplace=false)\n\nObtain the discrete pdf of the integer array x, optionally specifying the array of bins.\n\nThe result is normalized by default. If it is not normalized now, you can do so later by calling DeconvUtil.normalizepdf.\n\nLaplace correction means that at least one example is assumed in every bin, so that no bin has probability zero. This feature is disabled by default.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.DeconvUtil.fit_R","page":"API reference","title":"CherenkovDeconvolution.DeconvUtil.fit_R","text":"fit_R(y, x; bins_y, bins_x, normalize=true)\n\nEstimate the detector response matrix R, which empirically captures the transfer from the integer array y to the integer array x.\n\nR is normalized by default so that fit_pdf(x) == R * fit_pdf(y). If R is not normalized now, you can do so later calling DeconvUtil.normalizetransfer(R).\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.DeconvUtil.normalizetransfer","page":"API reference","title":"CherenkovDeconvolution.DeconvUtil.normalizetransfer","text":"normalizetransfer(R[; warn=true])\n\nNormalize each column in R to make a probability density function.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.DeconvUtil.normalizepdf","page":"API reference","title":"CherenkovDeconvolution.DeconvUtil.normalizepdf","text":"normalizepdf(array...; warn=true)\nnormalizepdf!(array...; warn=true)\n\nNormalize each array to a discrete probability density function.\n\nBy default, warn if coping with NaNs, Infs, or negative values.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.DeconvUtil.normalizepdf!","page":"API reference","title":"CherenkovDeconvolution.DeconvUtil.normalizepdf!","text":"normalizepdf(array...; warn=true)\nnormalizepdf!(array...; warn=true)\n\nNormalize each array to a discrete probability density function.\n\nBy default, warn if coping with NaNs, Infs, or negative values.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.DeconvUtil.polynomial_smoothing","page":"API reference","title":"CherenkovDeconvolution.DeconvUtil.polynomial_smoothing","text":"polynomial_smoothing([o = 2, warn = true])\n\nCreate a function object f -> smoothing(f) which smoothes its argument with a polynomial of order o. warn specifies if a warning is emitted when negative values returned by the smoothing are replaced by the average of neighboring values - a post-processing step proposed in [dagostini2010improved].\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#Developer-interface","page":"API reference","title":"Developer interface","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"The following list of methods is primarily intended for developers who wish to implement their own deconvolution methods, binnings, stepsizes, etc. If you do so, please file a pull request so that others can benefit from your work! More information on how to develop for this package is given in the Developer manual.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"DeconvolutionMethod\nDiscreteMethod\n\nBinning\nBinningDiscretizer\nbins\nencode\n\nStepsize\nOptimizedStepsize\ninitialize!\nvalue\n\ncheck_prior\ncheck_arguments\nLoneClassException\nrecover_estimate\n\nLabelSanitizer\nencode_labels\nencode_prior\ndecode_estimate","category":"page"},{"location":"api-reference/#CherenkovDeconvolution.Methods.DeconvolutionMethod","page":"API reference","title":"CherenkovDeconvolution.Methods.DeconvolutionMethod","text":"abstract type DeconvolutionMethod\n\nThe supertype of all deconvolution methods.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Methods.DiscreteMethod","page":"API reference","title":"CherenkovDeconvolution.Methods.DiscreteMethod","text":"abstract type DiscreteMethod <: DeconvolutionMethod\n\nThe supertype of all classical deconvolution methods which estimate the density function f from a transfer matrix R and an observed density g.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Binnings.Binning","page":"API reference","title":"CherenkovDeconvolution.Binnings.Binning","text":"abstract type Binning\n\nSupertype of all binning strategies for observable features.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Binnings.BinningDiscretizer","page":"API reference","title":"CherenkovDeconvolution.Binnings.BinningDiscretizer","text":"abstract type BinningDiscretizer <: AbstractDiscretizer\n\nSupertype of any clustering-based discretizer mapping from an n-dimensional space to a single cluster index dimension.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Binnings.bins","page":"API reference","title":"CherenkovDeconvolution.Binnings.bins","text":"bins(d::T) where T <: BinningDiscretizer\n\nReturn the bin indices of d.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#Discretizers.encode","page":"API reference","title":"Discretizers.encode","text":"encode(d::TreeDiscretizer, X_obs)\n\nDiscretize X_obs using the leaf indices in the decision tree of d as discrete values.\n\n\n\n\n\nencode(d::KMeansDiscretizer, X_obs)\n\nDiscretize X_obs using the cluster indices of d as discrete values.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Stepsizes.Stepsize","page":"API reference","title":"CherenkovDeconvolution.Stepsizes.Stepsize","text":"abstract type stepsize end\n\nAbstract supertype for step sizes in deconvolution.\n\nSee also: stepsize.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.OptimizedStepsizes.OptimizedStepsize","page":"API reference","title":"CherenkovDeconvolution.OptimizedStepsizes.OptimizedStepsize","text":"OptimizedStepsize(objective, decay)\n\nA step size that is optimized over an objective function. If decay=true, then the step sizes never increase.\n\nSee also: RunStepsize, LsqStepsize.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Stepsizes.initialize!","page":"API reference","title":"CherenkovDeconvolution.Stepsizes.initialize!","text":"initialize!(s, X_obs, X_trn, y_trn)\n\nPrepare the stepsize strategy s with the observed features in X_obs and the training set (X_trn, y_trn).\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Stepsizes.value","page":"API reference","title":"CherenkovDeconvolution.Stepsizes.value","text":"value(s, k, p, f, a)\n\nUse the Stepsize object s to compute a step size for iteration number k with the search direction p, the previous estimate f, and the previous step size a.\n\nSee also: ConstantStepsize, RunStepsize, LsqStepsize, ExpDecayStepsize, MulDecayStepsize.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Methods.check_prior","page":"API reference","title":"CherenkovDeconvolution.Methods.check_prior","text":"check_prior(f_0, n_bins)\n\nThrow meaningful exceptions if the input prior of a deconvolution run is defective.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Methods.check_arguments","page":"API reference","title":"CherenkovDeconvolution.Methods.check_arguments","text":"check_arguments(X_obs, X_trn, y_trn)\n\nThrow meaningful exceptions if the input data of a deconvolution run is defective.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Methods.LoneClassException","page":"API reference","title":"CherenkovDeconvolution.Methods.LoneClassException","text":"LoneClassException(label)\n\nAn exception thrown by check_arguments when only one class is in the training set.\n\nSee also: recover_estimate\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Methods.recover_estimate","page":"API reference","title":"CherenkovDeconvolution.Methods.recover_estimate","text":"recover_estimate(x::LoneClassException, n_bins=1)\n\nRecover a trivial deconvolution result from x, in which all bins are zero, except for the one that occured in the training set.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Methods.LabelSanitizer","page":"API reference","title":"CherenkovDeconvolution.Methods.LabelSanitizer","text":"LabelSanitizer(y_trn, n_bins=maximum(y_trn))\n\nA sanitizer that\n\nencodes labels and priors so that none of the resulting bins is empty.\ndecodes deconvolution results to recover the original (possibly empty) bins.\n\nSee also: encode_labels, encode_prior, decode_estimate.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#CherenkovDeconvolution.Methods.encode_labels","page":"API reference","title":"CherenkovDeconvolution.Methods.encode_labels","text":"encode_labels(s::LabelSanitizer, y_trn)\n\nEncode the labels y_trn so that all values from 1 to max(y_trn) occur.\n\nSee also: encode_prior, decode_estimate.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Methods.encode_prior","page":"API reference","title":"CherenkovDeconvolution.Methods.encode_prior","text":"encode_prior(s::LabelSanitizer, f_0)\n\nEncode the prior f_0 to be consistent with the encoded labels.\n\nSee also: encode_labels, decode_estimate.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#CherenkovDeconvolution.Methods.decode_estimate","page":"API reference","title":"CherenkovDeconvolution.Methods.decode_estimate","text":"decode_estimate(s::LabelSanitizer, f)\n\nRecover the original bins in a deconvolution result f after encoding the labels.\n\nSee also: encode_labels, encode_prior.\n\n\n\n\n\n","category":"function"},{"location":"#Manual","page":"Manual","title":"CherenkovDeconvolution.jl","text":"","category":"section"},{"location":"","page":"Manual","title":"Manual","text":"Deconvolution problems arise when the probability density function of a quantity is estimated even though this quantity cannot be measured directly. In this scenario, the density has to be inferred from related quantities which are measured instead.","category":"page"},{"location":"#Getting-started","page":"Manual","title":"Getting started","text":"","category":"section"},{"location":"","page":"Manual","title":"Manual","text":"CherenkovDeconvolution.jl can be installed through the Julia package manager. From the Julia REPL, type ] to enter the Pkg mode of the REPL. Then run","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"pkg> add CherenkovDeconvolution","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"To deconvolve an observed unlabeled data set X_obs, you need to configure the deconvolution method and to provide a labeled data set (X_trn, y_trn) for training. For instance, you can apply the DSEA+ algorithm with a naive Bayes classifier from ScikitLearn.jl as follows:","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"using CherenkovDeconvolution, ScikitLearn\n@sk_import naive_bayes : GaussianNB\n\nm = DSEA(GaussianNB())\nf_dsea = deconvolve(m, X_obs, X_trn, y_trn)","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"Classical deconvolution methods require a binning object to discretize the features in X_obs and X_trn into a single discrete dimension; in fact, this discretization is nothing but a clustering.","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"m = RUN(TreeBinning(10)) # use up to 10 clusters\nf_run = deconvolve(m, X_obs, X_trn, y_trn)","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"As you can see, all deconvolution methods are executed in the same way, through a call to the deconvolve function. Their configuration is further documented in the API reference.","category":"page"},{"location":"#Adaptive-stepsizes-in-DSEA","page":"Manual","title":"Adaptive stepsizes in DSEA+","text":"","category":"section"},{"location":"","page":"Manual","title":"Manual","text":"DSEA+ extends the original DSEA with an adaptively chosen stepsize between iterations. Here, both versions of DSEA are implemented as the same deconvolution method, DSEA, and only differ in their configuration. By default, a constant step size of one is used, but through the stepsize argument, we can easily specify adaptive stepsizes.","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"The most important one, RunStepsize, uses the objective function of RUN to determine the step size adaptively. In the following example, epsilon specifies the minimum Chi square distance between iterations; convergence is assumed if the distance drops below this threshold.","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"stepsize = RunStepsize(TreeBinning(10); decay=true)\ndsea = DSEA(GaussianNB(); K=100, epsilon=1e-6, stepsize=stepsize)\nf_dsea = deconvolve(dsea, X_data, X_train, y_train)","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"Another adaptive step size, based on a least-square objective, is the LsqStepsize. Two decaying step sizes are implemented in ExpDecayStepsize and MulDecayStepsize. The DEFAULT_STEPSIZE for all methods is a ConstantStepsize of one.","category":"page"},{"location":"#Inspection-of-intermediate-results","page":"Manual","title":"Inspection of intermediate results","text":"","category":"section"},{"location":"","page":"Manual","title":"Manual","text":"Iterative algorithms like DSEA, IBU, and RUN allow you to inspect their intermediate results of each iteration through the keyword argument inspect. This argument accept a Function object, which will be called in each iteration of the deconvolution algorithm. Depending on the algorithm, this Function object must have one of several signatures.","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"For DSEA and IBU, the inspect function has to have the following signature:","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"(f_k::Vector, k::Int, chi2s::Float64, alpha::Float64) -> Any","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"You do not have to stick to the argument names (f_k, k, etc) and you do not even have to specify the types of the arguments explicitly. However, these are the types that the parameters will have, so do not expect anything else. The return value of the inspect function is never used, so you can return any value, including nothing.","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"The first argument of inspect, f_k, refers to the intermediate result of the k-th iteration. chi2s is the Chi-Square distance between f_k and the previous estimate. This distance may be used to check convergence. alpha is the step size used in the k-th iteration.","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"The inspect signatures of RUN and PRUN are specified in their documentation. They are slightly different but the general inspection mechanism is the same. SVD does not have an inspection mechanism because this method only performs a single iteration.","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"In the following example, we store every piece of information in a DataFrame:","category":"page"},{"location":"","page":"Manual","title":"Manual","text":"using DataFrames\ndf = DataFrame(f=Vector{Float64}[], k=Int[], chi2s=Float64[], a=Float64[])\n\n# append everything to the DataFrame\ninspect_function = (f, k, chi2s, a) -> push!(df, [f, k, chi2s, a])\nm = DSEA(\n    GaussianNB();\n    K = 3,\n    inspect = inspect_function\n)\n\ndeconvolve(m, X_obs, X_trn, y_trn) # start the deconvolution","category":"page"}]
}