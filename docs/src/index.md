# [CherenkovDeconvolution.jl](@id Manual)

Deconvolution problems arise when the probability density function of a quantity is estimated even though this quantity cannot be measured directly. In this scenario, the density has to be inferred from related quantities which are measured instead.

## Getting started

CherenkovDeconvolution.jl can be installed through the Julia package manager. From the Julia REPL, type `]` to enter the Pkg mode of the REPL. Then run

```
pkg> add CherenkovDeconvolution
```

To deconvolve an observed unlabeled data set `X_obs`, you need to configure the deconvolution method and to provide a labeled data set `(X_trn, y_trn)` for training. For instance, you can apply the *DSEA+* algorithm with a naive Bayes classifier from [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) as follows:

```julia
using CherenkovDeconvolution, ScikitLearn
@sk_import naive_bayes : GaussianNB

m = DSEA(GaussianNB())
f_dsea = deconvolve(m, X_obs, X_trn, y_trn)
```

Classical deconvolution methods require a binning object to discretize the features in `X_obs` and `X_trn` into a single discrete dimension; in fact, this discretization is nothing but a clustering.

```julia
m = RUN(TreeBinning(10)) # use up to 10 clusters
f_run = deconvolve(m, X_obs, X_trn, y_trn)
```

You can choose to cluster prediction scores of a classifier instead of clustering the features. Clustering these scores with a [`KMeansBinning`](@ref) can be beneficial if you already know a well-performing classifier for your data which is not a decision tree.

```julia
@sk_import linear_model : LogisticRegression

clf = LogisticRegression()
m = RUN(KMeansBinning(10, ClassificationPreprocessor(clf)))
f_run_clf = deconvolve(m, X_obs, X_trn, y_trn)
```

As you can see, all deconvolution methods are executed in the same way, through a call to the `deconvolve` function. Their configuration is further documented in the [API reference](@ref).

## Adaptive stepsizes in DSEA+

DSEA+ extends the original DSEA with an adaptively chosen stepsize between iterations. Here, both versions of DSEA are implemented as the same deconvolution method, [`DSEA`](@ref), and only differ in their configuration. By default, a constant step size of one is used, but through the `stepsize` argument, we can easily specify adaptive stepsizes.

The most important one, [`RunStepsize`](@ref), uses the objective function of [`RUN`](@ref) to determine the step size adaptively. In the following example, `epsilon` specifies the minimum Chi square distance between iterations; convergence is assumed if the distance drops below this threshold.

```julia
stepsize = RunStepsize(TreeBinning(10); decay=true)
dsea = DSEA(GaussianNB(); K=100, epsilon=1e-6, stepsize=stepsize)
f_dsea = deconvolve(dsea, X_data, X_train, y_train)
```

Another adaptive step size, based on a least-square objective, is the [`LsqStepsize`](@ref). Two decaying step sizes are implemented in [`ExpDecayStepsize`](@ref) and [`MulDecayStepsize`](@ref). The [`DEFAULT_STEPSIZE`](@ref) for all methods is a [`ConstantStepsize`](@ref) of one.

## Inspection of intermediate results

Iterative algorithms like DSEA, IBU, and RUN allow you to inspect their intermediate results of each iteration through the keyword argument `inspect`. This argument accept a `Function` object, which will be called in each iteration of the deconvolution algorithm. Depending on the algorithm, this `Function` object must have one of several signatures.

For [`DSEA`](@ref) and [`IBU`](@ref), the `inspect` function has to have the following signature:

```julia
(f_k::Vector, k::Int, chi2s::Float64, alpha::Float64) -> Any
```

You do not have to stick to the argument names (`f_k`, `k`, etc) and you do not even have to specify the types of the arguments explicitly. However, these are the types that the parameters will have, so do not expect anything else. The return value of the `inspect` function is never used, so you can return any value, including `nothing`.

The first argument of `inspect`, `f_k`, refers to the intermediate result of the `k`-th iteration. `chi2s` is the Chi-Square distance between `f_k` and the previous estimate. This distance may be used to check convergence. `alpha` is the step size used in the `k`-th iteration.

The `inspect` signatures of [`RUN`](@ref) and [`PRUN`](@ref) are specified in their documentation. They are slightly different but the general inspection mechanism is the same. [`SVD`](@ref) does not have an inspection mechanism because this method only performs a single iteration.

In the following example, we store every piece of information in a `DataFrame`:

```julia
using DataFrames
df = DataFrame(f=Vector{Float64}[], k=Int[], chi2s=Float64[], a=Float64[])

# append everything to the DataFrame
inspect_function = (f, k, chi2s, a) -> push!(df, [f, k, chi2s, a])
m = DSEA(
    GaussianNB();
    K = 3,
    inspect = inspect_function
)

deconvolve(m, X_obs, X_trn, y_trn) # start the deconvolution
```

## Improving performance by pre-fitting

Computing the [Binnings](api-reference.html#Binnings) for the discrete deconvolution methods typically takes a considerable amount of time. You can speed up the deconvolution of multiple observations `X_obs_1, X_obs_2, ...` by computing these binnings only once with [`prefit`](@ref):

```julia
m = RUN(TreeBinning(10))
m_prefit = prefit(m, X_trn, y_trn) # a fitted copy; m remains unchanged!
f_1 = deconvolve(m_prefit, X_obs_1)
f_2 = deconvolve(m_prefit, X_obs_2)
...
```
