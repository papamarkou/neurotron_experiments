# %% Import packages

import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

from sim_setup import output_path

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load numerical output

tron_neuron1_error_loaded = []

for k in range(3):
    tron_neuron1_error_loaded.append(np.loadtxt(output_path.joinpath('q_assist_neuro1_'+str(k)+'_tron.csv'), delimiter=','))

tron_neuron10_error_loaded = []

for k in range(3):
    tron_neuron10_error_loaded.append(np.loadtxt(output_path.joinpath('q_assist_neuro10_'+str(k)+'_tron.csv'), delimiter=','))

# %% Set font size

fontsize = 13

# %% Set transparency

transparent = False

# %% Set y axis limits and ticks

ylims = [
    [-6.2, 1.2],
    [-11.2, 1.2]
]

yticks = [
    [-6, -5, -4, -3, -2, -1, 0, 1],
    [-11, -9, -7, -5, -3, -1, 1]
]

ylabels = [
    ['1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'],
    ['1e-11', '1e-9', '1e-7', '1e-5', '1e-3', '1e-1', '1e+1']
]

# %%

iterations = 4*(10**4)

beta_val = 0.05

theta_vals = [0, 0.5, 1]

# %%

save = True

xrange = range(1, iterations+1)

plt.figure(figsize=[8, 5])

for i in range(3):
    plt.plot(
        xrange,
        np.log10(tron_neuron1_error_loaded[i]),
        linewidth=2.,
        label=r'$\beta = {}$, $\theta_\ast$ = {}'.format(beta_val, theta_vals[i])
    )

plt.ylim(ylims[0])

plt.title("Neurotron (q=1)", fontsize=fontsize)

plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel(r'Averaged parameter error ($\log_{10}$ scale)', fontsize=fontsize)

plt.yticks(ticks=yticks[0], labels=ylabels[0], fontsize=fontsize)

plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5, ncol=1)

if save:
    plt.savefig(
        output_path.joinpath('q_assist_neuro1_tron.png'),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )

# %%

xrange = range(1, iterations+1)

plt.figure(figsize=[8, 5])

for i in range(3):
    plt.plot(
        xrange,
        np.log10(tron_neuron10_error_loaded[i]),
        linewidth=2.,
        label=r'$\beta = {}$, $\theta_\ast$ = {}'.format(beta_val, theta_vals[i])
    )

plt.ylim(ylims[1])

plt.title("Neurotron (q=10)", fontsize=fontsize)

plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel(r'Averaged parameter error ($\log_{10}$ scale)', fontsize=fontsize)

plt.yticks(ticks=yticks[1], labels=ylabels[1], fontsize=fontsize)

plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5, ncol=1)

if save:
    plt.savefig(
        output_path.joinpath('q_assist_neuro10_tron.png'),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=transparent,
        bbox_inches='tight',
        pad_inches=0.1
    )
