# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim05_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim05_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim05_setup['sample_data'])

# %% Run neurotron for simulation 5

tron_error, sgd_error = neurotron.run(
    sim05_setup['filterlist'],
    sim05_setup['dlist'],
    sim05_setup['boundlist'],
    sim05_setup['betalist'],
    sim05_setup['etalist'],
    sim05_setup['blist'],
    sim05_setup['width'],
    sim05_setup['num_iters'],
    run_sgd=sim05_setup['run_sgd']
)

# %% Save output of simulation 5

np.savetxt(output_path.joinpath(sim05_setup['name']+'_tron.csv'), np.squeeze(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim05_setup['name']+'_sgd.csv'), np.squeeze(sgd_error), delimiter=',')
