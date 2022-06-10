# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim06_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim06_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim06_setup['sample_data'])

# %% Run neurotron

tron_error, sgd_error = neurotron.run(
    sim06_setup['filterlist'],
    sim06_setup['dlist'],
    sim06_setup['boundlist'],
    sim06_setup['betalist'],
    sim06_setup['etalist_tron'],
    sim06_setup['blist'],
    sim06_setup['width'],
    sim06_setup['num_iters'],
    etalist_sgd=sim06_setup['etalist_sgd']
)

# %% Save output

np.savetxt(output_path.joinpath(sim06_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim06_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
