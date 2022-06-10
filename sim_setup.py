# %% Import packages

import numpy as np

from pathlib import Path

# %% Set output path

output_path = Path().joinpath('output')

# %% Setup for simulation 1: data ~ normal(mu=0, sigma=1), varying theta_{*}

sim01_setup = {
    'sample_data' : lambda s : np.random.normal(loc=0.0, scale=1.0, size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [100 for _ in range(7)], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5 for _ in range(7)], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 1,
    'name' : 'sim01'
}

# %% Setup for simulation 2: data ~ normal(mu=0, sigma=3), varying theta_{*}

sim02_setup = {
    'sample_data' : lambda s : np.random.normal(loc=0.0, scale=3.0, size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [50 for _ in range(7)], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5 for _ in range(7)], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 2,
    'name' : 'sim02'
}

# %% Setup for simulation 3: data ~ Laplace(loc=0, scale=2), varying theta_{*}

sim03_setup = {
    'sample_data' : lambda s : np.random.laplace(loc=0.0, scale=2.0, size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [50 for _ in range(7)], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5 for _ in range(7)], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 3,
    'name' : 'sim03'
}

# %% Setup for simulation 4: data ~ student(df=4), varying theta_{*}

sim04_setup = {
    'sample_data' : lambda s : np.random.standard_t(4., size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [100 for _ in range(7)], # n: input dimension
    'boundlist' : [0, 0.125, 0.25, 0.5, 1., 2., 4.], # theta_{*}
    'betalist' : [0.5 for _ in range(7)], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 4,
    'name' : 'sim04'
}

# %% Setup for simulation 5: data ~ normal(mu=0, sigma=1), varying beta

sim05_setup = {
    'sample_data' : lambda s : np.random.normal(loc=0.0, scale=1.0, size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [100 for _ in range(7)], # n: input dimension
    'boundlist' : [0.25 for _ in range(7)], # theta_{*}
    'betalist' : [0., 0.005, 0.05, 0.1, 0.2, 0.5, 0.9], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 5,
    'name' : 'sim05'
}

# %% Setup for simulation 6: data ~ normal(mu=0, sigma=3), varying beta

sim06_setup = {
    'sample_data' : lambda s : np.random.normal(loc=0.0, scale=3.0, size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [50 for _ in range(7)], # n: input dimension
    'boundlist' : [0.25 for _ in range(7)], # theta_{*}
    'betalist' : [0., 0.005, 0.05, 0.1, 0.2, 0.5, 0.9], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 6,
    'name' : 'sim06'
}

# %% Setup for simulation 7: data ~ Laplace(loc=0, scale=2), varying beta

sim07_setup = {
    'sample_data' : lambda s : np.random.laplace(loc=0.0, scale=2.0, size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [50 for _ in range(7)], # n: input dimension
    'boundlist' : [0.25 for _ in range(7)], # theta_{*}
    'betalist' : [0., 0.005, 0.05, 0.1, 0.2, 0.5, 0.9], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 7,
    'name' : 'sim07'
}

# %% Setup for simulation 8: data ~ student(df=4), varying beta

sim08_setup = {
    'sample_data' : lambda s : np.random.standard_t(4., size=s),
    'filterlist' : [25 for _ in range(7)], # r: filter size
    'dlist' : [100 for _ in range(7)], # n: input dimension
    'boundlist' : [0.25 for _ in range(7)], # theta_{*}
    'betalist' : [0., 0.005, 0.05, 0.1, 0.2, 0.5, 0.9], # beta
    'etalist_tron' : [0.0001 for _ in range(7)], # eta: learning rate for NeuroTron
    'blist' : [16 for _ in range(7)], # b
    'width' : 10, # k: width
    'num_iters' : 40000,
    'etalist_sgd' : [0.0001 for _ in range(7)], # eta: learning rate for SGD
    'seed' : 8,
    'name' : 'sim08'
}
