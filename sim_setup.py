# %% Import packages

from pathlib import Path

# %% Set output path

output_path = Path().joinpath('output')

# %% Setup for simulation 1

sim01_setup = {
    'filterlist' : [25], # r: filter size
    'dlist' : [100], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 1., 2., 4.], # theta_{*} 
    'betalist' : [0.5], # beta
    'etalist' : [0.0001], # eta: learning rate
    'blist' : [16], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'name' : 'sim01'
}
