# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim01_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim01_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim01_setup['sample_data'])

# %% Run neurotron

tron_error, sgd_error = neurotron.run(
    sim01_setup['filterlist'],
    sim01_setup['dlist'],
    sim01_setup['boundlist'],
    sim01_setup['betalist'],
    sim01_setup['etalist_tron'],
    sim01_setup['blist'],
    sim01_setup['width'],
    sim01_setup['num_iters'],
    etalist_sgd=sim01_setup['etalist_sgd']
)

# %% Save output

np.savetxt(output_path.joinpath(sim01_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim01_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
