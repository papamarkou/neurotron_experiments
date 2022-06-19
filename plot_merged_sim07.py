# %% Import packages

import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

from sim_setup import output_path, sim07_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load numerical output

tron_error_loaded = np.loadtxt(output_path.joinpath(sim07_setup['name']+'_tron.csv'), delimiter=',')
sgd_error_loaded = np.loadtxt(output_path.joinpath(sim07_setup['name']+'_sgd.csv'), delimiter=',')

# %% Set font size

fontsize = 13

# %% Set transparency

transparent = False

# %% Set y axis limits and ticks

ylims = [
    [-16.2, 2.2],
    [-4.2, 1.2],
    [-4.2, 1.2],
    [-4.2, 1.2],
    [-4.2, 1.2],
    [-4.2, 1.2],
    [-4.2, 1.2]
]

yticks = [
    [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1],
    [-4, -3, -2, -1, 0, 1]
]

ylabels = [
    ['1e-16', '1e-14', '1e-12', '1e-10', '1e-8', '1e-6', '1e-4', '1e-2', '1e-0', '1e+2'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1']
]

# %% Selection of theta values to plot

beta_vals = [1, 3, 4, 5, 6]

# %%

save = True

xrange = range(1, tron_error_loaded.shape[0]+1)

labels = ['Neurotron', 'SGD']

fig, axes = plt.subplots(nrows=len(beta_vals), ncols=1, sharex=True, figsize=(8, 18))

plt.subplots_adjust(hspace = 0.15)

for i in range(len(beta_vals)):
    axes[i].plot(
        xrange,
        np.log10(tron_error_loaded[:, beta_vals[i]]),
        linewidth=2.,
        label=labels[0]
    )

    axes[i].plot(
        xrange,
        np.log10(sgd_error_loaded[:, beta_vals[i]]),
        linewidth=2.,
        label=labels[1]
    )

    axes[i].set_ylim(ylims[beta_vals[i]])

    axes[i].set_title(
        r'$\beta$ = {}'.format(sim07_setup['betalist'][beta_vals[i]]), y=1.0, pad=-23, fontsize=fontsize
    )

    axes[i].set_yticks(yticks[beta_vals[i]])
    axes[i].set_yticklabels(ylabels[beta_vals[i]], fontsize=fontsize)

    axes[i].legend(labels=labels, loc='upper right', ncol=1, fontsize=fontsize, frameon=False)

xticks = np.linspace(0, 40000, num=9)
xticklabels = [str(round(i)) for i in xticks]

axes[4].set_xticks(xticks)
axes[4].set_xticklabels(xticklabels, rotation=0, fontsize=fontsize)

if save:
    plt.savefig(
        output_path.joinpath(
            sim07_setup['name']+'_tron_vs_sgd_merged_beta_vals.png'
        ),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )
