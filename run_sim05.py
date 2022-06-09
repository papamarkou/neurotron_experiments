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

# %% Run neurotron

tron_error, sgd_error = neurotron.run(
    sim05_setup['filterlist'],
    sim05_setup['dlist'],
    sim05_setup['boundlist'],
    sim05_setup['betalist'],
    sim05_setup['etalist_tron'],
    sim05_setup['blist'],
    sim05_setup['width'],
    sim05_setup['num_iters'],
    etalist_sgd=sim05_setup['etalist_sgd']
)

# %% Save output

np.savetxt(output_path.joinpath(sim05_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim05_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
