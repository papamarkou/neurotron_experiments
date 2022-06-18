# %% Import packages

import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

from sim_setup import output_path, sim01_setup, sim02_setup, sim03_setup, sim04_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load numerical output

tron_error_loaded = []

for sim_setup in [sim01_setup, sim02_setup, sim03_setup, sim04_setup]:
    tron_error_loaded.append(np.loadtxt(output_path.joinpath(sim_setup['name']+'_tron.csv'), delimiter=','))

# %% Set font size

fontsize = 13

# %% Set transparency

transparent = False

# %% Set y axis limits and ticks

ylims = [-16.2, 2.2],

yticks = [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2]

ylabels = ['1e-16', '1e-14', '1e-12', '1e-10', '1e-8', '1e-6', '1e-4', '1e-2', '1e-0', '1e+2']

leg_labels = [
    r'Normal data ($\sigma=1$)',
    r'Normal data ($\sigma=3$)',
    r'Laplace data ($scale=2$)',
    r't-distributed data ($df=4$)'

]

# %% Generate and save NeuroTron-vs-SGD figure

save = True

plt.figure(figsize=[8, 4])

for i in range(4):

    xrange = range(1, tron_error_loaded[0].shape[0]+1)

    labels = ['Neurotron', 'SGD']

    plt.plot(
        xrange,
        np.log10(tron_error_loaded[i][:, 0]),
        linewidth=2.,
        label=leg_labels[i]
    )

plt.ylim([-15.4, 1.4])

plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

plt.yticks(
    ticks=[-15, -13, -11, -9, -7, -5, -3, -1, 1],
    labels=['1e-15', '1e-13', '1e-11', '1e-9', '1e-7', '1e-5', '1e-3', '1e-1', '1e+1'],
    fontsize=fontsize
)

plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5, ncol=1)

# %%

if save:
    plt.savefig(
        output_path.joinpath('tron_theta_no_attack.png'),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )
