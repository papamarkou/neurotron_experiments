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

ylims = [
    # [-16.2, 2.2],
    [-4.2, 1.2],
    [-4.2, 1.2],
    [-4.2, 1.2],
    [-4.2, 1.2]
]

yticks = [
    # [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1]
]

ylabels = [
    # ['1e-16', '1e-14', '1e-12', '1e-10', '1e-8', '1e-6', '1e-4', '1e-2', '1e-0', '1e+2'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1']
]

# %% Selection of theta values to plot

theta_vals = [0, 1, 2, 3, 4, 5, 6]

# %%

save = False

xrange = range(1, tron_error_loaded[0].shape[0]+1)

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 18))

plt.subplots_adjust(hspace = 0.15)

for i in range(4):
    for j in range(len(theta_vals)):
        axes[i].plot(
            xrange,
            np.log10(tron_error_loaded[i][:, theta_vals[j]]),
            linewidth=2. # ,
            # label=labels[0]
        )

    axes[i].set_ylim(ylims[theta_vals[i]])

    axes[i].set_yticks(yticks[theta_vals[i]])
    axes[i].set_yticklabels(ylabels[theta_vals[i]], fontsize=fontsize)

if save:
    plt.savefig(
        output_path.joinpath('all_sims_tron_merged_theta_vals.png'),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )
