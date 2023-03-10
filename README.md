# RESSPECT_pipeline_paper
This repository holds necessary tools to reproduce results from Ishida et al., 2023.


### Data

In this work we used the PLAsTiCC zenodo data: https://zenodo.org/record/2539456#.Y--dm9LMJhE

### Pre-processing

1) Fit the entire PLAsTiCC test sample using SALT2 using the [code from Malz et al., 2023](https://github.com/COINtoolbox/RESSPECT_metric/blob/main/code/01_SALT2_fit.py).

2) Create validation, test and pool samples using [notebook 01](https://github.com/COINtoolbox/RESSPECT_pipeline_paper/blob/main/code/01_make_samples.ipynb)

3) Use the [cosmology fit script](https://github.com/COINtoolbox/RESSPECT_pipeline_paper/blob/main/code/get_cosmo_posterior.py) to access the best/worst possible cosmological results from the validation sample
