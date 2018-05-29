[![Build Status](https://travis-ci.org/mirkobunse/CherenkovDeconvolution.jl.svg?branch=master)](https://travis-ci.org/mirkobunse/CherenkovDeconvolution.jl)

# CherenkovDeconvolution.jl

Deconvolution methods for Cherenkov astronomy and other use cases in experimental physics.


### Getting Started

Install the package by cloning out this repository with the Julia package manager.
Also consider installing
[ScikitLearn.jl](http://scikitlearnjl.readthedocs.io/en/latest/quickstart/),
with which you can easily set up the classifiers used in DSEA.

      Pkg.clone("git://github.com/mirkobunse/CherenkovDeconvolution.jl.git")
      Pkg.add("ScikitLearn")

You can use the deconvolution methods, right away.

      import ScikitLearn, CherenkovDeconvolution
      
      # 
      # configure the classifier you want to use
      # 
      # (Sklearn is a sub-module of CherenkovDeconvolution, which is dynamically loaded if
      # ScikitLearn is available)
      # 
      trainpredict = Sklearn.train_and_predict_proba("GaussianNB") # Naive Bayes
      
      # read your DataFrames for deconvolution and training
      data = ...
      train = ...
      y = :y # assuming this is your target quantity
      
      # perform deconvolution with DSEA
      CherenkovDeconvolution.dsea(data, train, y, trainpredict)

