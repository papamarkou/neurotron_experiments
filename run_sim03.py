# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim03_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim03_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim03_setup['sample_data'])

# %% Run neurotron for simulation 1

tron_error, sgd_error = neurotron.run(
    sim03_setup['filterlist'],
    sim03_setup['dlist'],
    sim03_setup['boundlist'],
    sim03_setup['betalist'],
    sim03_setup['etalist'],
    sim03_setup['blist'],
    sim03_setup['width'],
    sim03_setup['num_iters'],
    run_sgd=sim03_setup['run_sgd']
)

# %% Save output of simulation 1

np.savetxt(output_path.joinpath(sim03_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim03_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
