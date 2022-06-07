# %% Import packages

import numpy as np

from pathlib import Path

# %% Set output path

output_path = Path().joinpath('output')

# %% Setup for simulation 1

num_runs = 7

sim01_setup = {
    'sample_data' : lambda s : np.random.normal(loc=0.0, scale=1.0, size=s),
    'filterlist' : [25 for _ in range(num_runs)], # r: filter size
    'dlist' : [100 for _ in range(num_runs)], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5 for _ in range(num_runs)], # beta
    'etalist' : [0.0001 for _ in range(num_runs)], # eta: learning rate
    'blist' : [16 for _ in range(num_runs)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'run_sgd' : True,
    'seed' : 1,
    'name' : 'sim01'
}
