# %% Import packages

import numpy as np

from pathlib import Path

# %% Set output path

output_path = Path().joinpath('output')

# %% Setup for simulation 1

sim01_setup = {
    'sample_data' : lambda s : np.random.normal(loc=0.0, scale=1.0, size=s),
    'filterlist' : [25], # r: filter size
    'dlist' : [100], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5], # beta
    'etalist' : [0.0001], # eta: learning rate
    'blist' : [16], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'run_sgd' : True,
    'seed' : 1,
    'name' : 'sim01'
}

# %% Setup for simulation 2

sim02_setup = {
    'sample_data' : lambda s : np.random.laplace(loc=0.0, scale=2.0, size=s),
    'filterlist' : [25], # r: filter size
    'dlist' : [50], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5], # beta
    'etalist' : [0.0001], # eta: learning rate
    'blist' : [16], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'run_sgd' : True,
    'seed' : 2,
    'name' : 'sim02'
}

# %% Setup for simulation 3

sim03_setup = {
    'sample_data' : lambda s : np.random.standard_t(3., size=s),
    'filterlist' : [25], # r: filter size
    'dlist' : [100], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5], # beta
    'etalist' : [0.0001], # eta: learning rate
    'blist' : [16], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'run_sgd' : True,
    'seed' : 3,
    'name' : 'sim03'
}
