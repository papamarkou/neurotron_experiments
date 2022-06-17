# %% Import packages

import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

from sim_setup import output_path, sim05_setup, sim06_setup, sim07_setup, sim08_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load numerical output

tron_error_loaded = []

for sim_setup in [sim05_setup, sim06_setup, sim07_setup, sim08_setup]:
    tron_error_loaded.append(np.loadtxt(output_path.joinpath(sim_setup['name']+'_tron.csv'), delimiter=','))

# %% Set font size

fontsize = 13

# %% Set transparency

transparent = False

# %% Set y axis limits and ticks

ylims = [
    [-16.2, 2.2],
    [-4.4, 1.2],
    [-4.4, 1.2],
    [-4.4, 1.2],
    [-4.4, 1.2],
    [-4.4, 1.2],
    [-4.4, 1.2]
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

# %% Set sub-plots titles

titles = [
    r'Normal data ($\sigma=1$)',
    r'Normal data ($\sigma=3$)',
    r'Laplace data ($scale=2$)',
    r't-distributed data ($df=4$)'
]

# %% Set line labels

sim01_theta_labels = [
    r'$\beta$ = {}'.format(sim05_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][6])
]

sim02_theta_labels = [
    r'$\beta$ = {}'.format(sim06_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][6])
]

sim03_theta_labels = [
    r'$\beta$ = {}'.format(sim07_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][6])
]

sim04_theta_labels = [
    r'$\beta$ = {}'.format(sim08_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][6])
]

theta_labels = [sim01_theta_labels, sim02_theta_labels, sim03_theta_labels, sim04_theta_labels]

# %% Selection of theta values to plot

theta_vals = [1, 2, 3, 4, 5, 6]

# %%

save = True

xrange = range(1, tron_error_loaded[0].shape[0]+1)

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 18))

plt.subplots_adjust(hspace = 0.15)

for i in range(4):
    for j in range(len(theta_vals)):
        axes[i].plot(
            xrange,
            np.log10(tron_error_loaded[i][:, theta_vals[j]]),
            linewidth=2.,
            label=theta_labels[i][theta_vals[j]]
        )

    axes[i].set_ylim(ylims[theta_vals[j]])

    axes[i].set_title(titles[i].format(sim05_setup['betalist'][theta_vals[j]]), y=1.0, fontsize=fontsize)

    axes[i].set_yticks(yticks[theta_vals[j]])
    axes[i].set_yticklabels(ylabels[theta_vals[j]], fontsize=fontsize)

    axes[i].legend(loc='upper right', ncol=2, fontsize=fontsize, frameon=False)

if save:
    plt.savefig(
        output_path.joinpath('all_sims_tron_merged_theta_vals.png'),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )
