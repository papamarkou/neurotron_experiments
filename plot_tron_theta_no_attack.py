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

ylims = [-16.2, 2.2]

# %%

save = True

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(16, 8))

plt.subplots_adjust(hspace=0.15, wspace=0.03)

xrange = range(1, tron_error_loaded[0].shape[0]+1)

for i in range(2):
    axes[0, 0].plot(
        xrange,
        np.log10(tron_error_loaded[0][:, i]),
        linewidth=2. # ,
        # label=labels[0]
    )

axes[0, 0].set_title(r'Normal data ($\sigma=1$)', fontsize=fontsize)

axes[0, 0].set_ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)
axes[1, 0].set_ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

axes[0, 0].set_ylim([-16.2, 2.2])

axes[0, 0].set_yticks([-16, -14, -12, -10, -8, -6, -4, -2, 0, 2])

axes[0, 0].set_yticklabels(
    ['1e-16', '1e-14', '1e-12', '1e-10', '1e-8', '1e-6', '1e-4', '1e-2', '1e-0', '1e+2'],
    rotation=0,
    fontsize=fontsize
)

for i in range(2):
    axes[1, 0].plot(
        xrange,
        np.log10(tron_error_loaded[1][:, i]),
        linewidth=2. # ,
        # label=labels[0]
    )

axes[1, 0].set_title(r'Normal data ($\sigma=3$)', fontsize=fontsize)

for i in range(2):
    axes[0, 1].plot(
        xrange,
        np.log10(tron_error_loaded[2][:, i]),
        linewidth=2. # ,
        # label=labels[0]
    )

axes[0, 1].set_title(r'Laplace data ($scale=2$)', fontsize=fontsize)

for i in range(2):
    axes[1, 1].plot(
        xrange,
        np.log10(tron_error_loaded[3][:, i]),
        linewidth=2. # ,
        # label=labels[0]
    )

axes[1, 1].set_title(r't-distributed data ($df=4$)', fontsize=fontsize)

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
