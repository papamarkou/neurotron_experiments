# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim01_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(seed=1)

# %% Set data sampling distribution

sample_data = lambda s : np.random.normal(loc=0.0, scale=1.0, size=s)

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data)

# %% Run neurotron for figure 1

tron_error, sgd_error = neurotron.run(
    sim01_setup['filterlist'],
    sim01_setup['dlist'],
    sim01_setup['boundlist'],
    sim01_setup['betalist'],
    sim01_setup['etalist'],
    sim01_setup['blist'],
    sim01_setup['width'],
    sim01_setup['num_iters']
)

# %% Savie output of simulation 1

np.savetxt(output_path.joinpath(sim01_setup['name']+'.csv'), np.squeeze(tron_error), delimiter=',')
