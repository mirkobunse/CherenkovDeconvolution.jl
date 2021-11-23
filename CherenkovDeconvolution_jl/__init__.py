def __jl():
    from julia import Main
    return Main

def install(version="main"):
    import julia
    julia.install()
    __jl().eval(f"""
        using Pkg
        Pkg.activate(".")
        Pkg.add(url="https://github.com/mirkobunse/CherenkovDeconvolution.jl.git", rev="{version}")
    """)

def initialize():
    __jl().eval("""
        import CherenkovDeconvolution.Methods: deconvolve, DSEA, IBU, PRUN, RUN, SVD
    """)

def deconvolve(m, X_obs, X_trn, y_trn):
    return __jl().deconvolve(m, X_obs, X_trn, y_trn)
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
