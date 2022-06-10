# %% Import packages

import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

from sim_setup import output_path, sim04_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load numerical output

tron_error_loaded = np.loadtxt(output_path.joinpath(sim04_setup['name']+'_tron.csv'), delimiter=',')
sgd_error_loaded = np.loadtxt(output_path.joinpath(sim04_setup['name']+'_sgd.csv'), delimiter=',')

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

# %% Generate and save NeuroTron-vs-SGD figure

save = True

for i in range(tron_error_loaded.shape[1]):
    plt.figure(figsize=[8, 5])

    xrange = range(1, tron_error_loaded.shape[0]+1)

    labels = ['Neurotron', 'SGD']

    plt.plot(
        xrange,
        np.log10(tron_error_loaded[:, i]),
        linewidth=2.,
        label=labels[0]
    )

    plt.plot(
        xrange,
        np.log10(sgd_error_loaded[:, i]),
        linewidth=2.,
        label=labels[1]
    )

    plt.ylim(ylims[i])

    plt.title(r'Student\'s t data ($df=4$), $\theta_\ast$ = {}'.format(sim04_setup['boundlist'][i]))

    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

    xtickstep = 5000
    xticks = range(0, sgd_error_loaded.shape[0]+xtickstep, xtickstep)
    plt.xticks(ticks=xticks, fontsize=fontsize)

    plt.yticks(ticks=yticks[i], labels=ylabels[i], fontsize=fontsize)

    leg = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5, ncol=2)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.)

    if save:
        plt.savefig(
            output_path.joinpath(
                sim04_setup['name']+'_tron_vs_sgd_theta_val'+str(i+1).zfill(len(str(tron_error_loaded.shape[1])))+'.png'
            ),
            dpi=300,
            pil_kwargs={'quality': 100},
            transparent=transparent,
            bbox_inches='tight',
            pad_inches=0.1
        )
