# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim04_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim04_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim04_setup['sample_data'])

# %% Run neurotron

tron_error, sgd_error = neurotron.run(
    sim04_setup['filterlist'],
    sim04_setup['dlist'],
    sim04_setup['boundlist'],
    sim04_setup['betalist'],
    sim04_setup['etalist'],
    sim04_setup['blist'],
    sim04_setup['width'],
    sim04_setup['num_iters'],
    run_sgd=sim04_setup['run_sgd']
)

# %% Save output

np.savetxt(output_path.joinpath(sim04_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim04_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
