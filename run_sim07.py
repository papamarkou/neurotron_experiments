# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim07_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim07_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim07_setup['sample_data'])

# %% Run neurotron for simulation 7

tron_error, sgd_error = neurotron.run(
    sim07_setup['filterlist'],
    sim07_setup['dlist'],
    sim07_setup['boundlist'],
    sim07_setup['betalist'],
    sim07_setup['etalist'],
    sim07_setup['blist'],
    sim07_setup['width'],
    sim07_setup['num_iters'],
    run_sgd=sim07_setup['run_sgd']
)

# %% Save output of simulation 7

np.savetxt(output_path.joinpath(sim07_setup['name']+'_tron.csv'), np.squeeze(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim07_setup['name']+'_sgd.csv'), np.squeeze(sgd_error), delimiter=',')
