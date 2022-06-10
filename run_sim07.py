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

# %% Run neurotron

tron_error, sgd_error = neurotron.run(
    sim07_setup['filterlist'],
    sim07_setup['dlist'],
    sim07_setup['boundlist'],
    sim07_setup['betalist'],
    sim07_setup['etalist_tron'],
    sim07_setup['blist'],
    sim07_setup['width'],
    sim07_setup['num_iters'],
    etalist_sgd=sim07_setup['etalist_sgd']
)

# %% Save output

np.savetxt(output_path.joinpath(sim07_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim07_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
