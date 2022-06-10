# %% Import packages

import numpy as np

from pathlib import Path

from neurotron import NeuroTron

from sim_setup import output_path, sim08_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Set the seed

np.random.seed(sim08_setup['seed'])

# %% Instantiate NeuroTron

neurotron = NeuroTron(sample_data=sim08_setup['sample_data'])

# %% Run neurotron

tron_error, sgd_error = neurotron.run(
    sim08_setup['filterlist'],
    sim08_setup['dlist'],
    sim08_setup['boundlist'],
    sim08_setup['betalist'],
    sim08_setup['etalist_tron'],
    sim08_setup['blist'],
    sim08_setup['width'],
    sim08_setup['num_iters'],
    etalist_sgd=sim08_setup['etalist_sgd']
)

# %% Save output

np.savetxt(output_path.joinpath(sim08_setup['name']+'_tron.csv'), np.transpose(tron_error), delimiter=',')
np.savetxt(output_path.joinpath(sim08_setup['name']+'_sgd.csv'), np.transpose(sgd_error), delimiter=',')
