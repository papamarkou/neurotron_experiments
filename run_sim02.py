# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim02_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim02_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim02_setup['sample_data'])

# %% Run neurotron

tron_error, sgd_error = neurotron.run(
    sim02_setup['filterlist'],
    sim02_setup['dlist'],
    sim02_setup['boundlist'],
    sim02_setup['betalist'],
    sim02_setup['etalist'],
    sim02_setup['blist'],
    sim02_setup['width'],
    sim02_setup['num_iters'],
    run_sgd=sim02_setup['run_sgd']
)

# %% Save output

np.savetxt(output_path.joinpath(sim02_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim02_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
