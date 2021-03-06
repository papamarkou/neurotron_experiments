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
    # [-16.2, 2.2],
    [-4.3, 1.2],
    [-4.3, 1.2],
    [-4.3, 1.2],
    [-4.3, 1.2]
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

# %% Set sub-plots titles

titles = [
    r'Normal data ($\sigma=1$)',
    r'Normal data ($\sigma=3$)',
    r'Laplace data ($scale=2$)',
    r't-distributed data ($df=4$)'
]

# %% Set line labels

sim01_beta_labels = [
    r'$\beta$ = {}'.format(sim05_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim05_setup['betalist'][6])
]

sim02_beta_labels = [
    r'$\beta$ = {}'.format(sim06_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim06_setup['betalist'][6])
]

sim03_beta_labels = [
    r'$\beta$ = {}'.format(sim07_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim07_setup['betalist'][6])
]

sim04_beta_labels = [
    r'$\beta$ = {}'.format(sim08_setup['betalist'][0]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][1]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][2]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][3]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][4]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][5]),
    r'$\beta$ = {}'.format(sim08_setup['betalist'][6])
]

beta_labels = [sim01_beta_labels, sim02_beta_labels, sim03_beta_labels, sim04_beta_labels]

# %% Selection of beta values to plot

beta_vals = [1, 2, 3, 4, 5, 6]

# %%

save = True

xrange = range(1, tron_error_loaded[0].shape[0]+1)

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 18))

plt.subplots_adjust(hspace = 0.15)

for i in range(4):
    for j in range(len(beta_vals)):
        axes[i].plot(
            xrange,
            np.log10(tron_error_loaded[i][:, beta_vals[j]]),
            linewidth=2.,
            label=beta_labels[i][beta_vals[j]]
        )

    axes[i].set_ylim(ylims[i])

    axes[i].set_title(titles[i].format(sim05_setup['betalist'][beta_vals[j]]), y=1.0, fontsize=fontsize)

    axes[i].set_yticks(yticks[i])
    axes[i].set_yticklabels(ylabels[i], fontsize=fontsize)

    axes[i].legend(loc='upper right', ncol=2, fontsize=fontsize, frameon=False)

xticks = np.linspace(0, 40000, num=9)
xticklabels = [str(round(i)) for i in xticks]

axes[3].set_xticks(xticks)
axes[3].set_xticklabels(xticklabels, rotation=0, fontsize=fontsize)

if save:
    plt.savefig(
        output_path.joinpath('all_sims_tron_merged_beta_vals.png'),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )
