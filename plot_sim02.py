# %% Import packages

import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

from sim_setup import output_path, sim02_setup

# %% Create output path if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load numerical output of simulation 2

tron_error_loaded = np.loadtxt(output_path.joinpath(sim02_setup['name']+'_tron.csv'), delimiter=',')
sgd_error_loaded = np.loadtxt(output_path.joinpath(sim02_setup['name']+'_sgd.csv'), delimiter=',')

# %% Set font size

fontsize = 13

# %% Generate and save NeuroTron figure for simulation 2

plt.figure(figsize=[8, 5])

xrange = range(1, tron_error_loaded.shape[0]+1)

labels = [r'$\theta_\ast$ = {}'.format(sim02_setup['boundlist'][i]) for i in range(tron_error_loaded.shape[1])]

for i in range(tron_error_loaded.shape[1]):
    plt.plot(
        xrange,
        np.log10(tron_error_loaded[:, i]),
        linewidth=2.,
        label=labels[i]
    )

plt.ylim([-17, 2])

plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

xtickstep = 5000
xticks = range(0, tron_error_loaded.shape[0]+xtickstep, xtickstep)
plt.xticks(ticks=xticks, fontsize=fontsize)

yticks = [-15, -10, -5, 0]
plt.yticks(ticks=yticks, labels=['1e-15', '1e-10', '1e-5', '1e-0'], fontsize=fontsize)

leg = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5)

for legobj in leg.legendHandles:
    legobj.set_linewidth(3.)

plt.savefig(
    output_path.joinpath(sim02_setup['name']+'_tron.png'),
    dpi=300,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

# %% Generate and save SGD figure for simulation 2

plt.figure(figsize=[8, 5])

xrange = range(1, sgd_error_loaded.shape[0]+1)

labels = [r'$\theta_\ast$ = {}'.format(sim02_setup['boundlist'][i]) for i in range(sgd_error_loaded.shape[1])]

for i in range(sgd_error_loaded.shape[1]):
    plt.plot(
        xrange,
        np.log10(sgd_error_loaded[:, i]),
        linewidth=2.,
        label=labels[i]
    )

plt.ylim([-17, 2])

plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

xtickstep = 5000
xticks = range(0, sgd_error_loaded.shape[0]+xtickstep, xtickstep)
plt.xticks(ticks=xticks, fontsize=fontsize)

yticks = [-15, -10, -5, 0]
plt.yticks(ticks=yticks, labels=['1e-15', '1e-10', '1e-5', '1e-0'], fontsize=fontsize)

leg = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5)

for legobj in leg.legendHandles:
    legobj.set_linewidth(3.)

plt.savefig(
    output_path.joinpath(sim02_setup['name']+'_sgd.png'),
    dpi=300,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

# %% Generate and save zoomed in NeuroTron figure for simulation 2

plt.figure(figsize=[8, 5])

xrange = range(1, tron_error_loaded.shape[0]+1)

labels = [r'$\theta_\ast$ = {}'.format(sim02_setup['boundlist'][i]) for i in range(tron_error_loaded.shape[1])]

for i in range(tron_error_loaded.shape[1]):
    plt.plot(
        xrange,
        np.log10(tron_error_loaded[:, i]),
        linewidth=2.,
        label=labels[i]
    )

plt.ylim([-4.2, 0.2])   

plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

xtickstep = 5000
xticks = range(0, tron_error_loaded.shape[0]+xtickstep, xtickstep)
plt.xticks(ticks=xticks, fontsize=fontsize)

yticks = [-4, -3, -2, -1, 0]
plt.yticks(ticks=yticks, labels=['1e-4', '1e-3', '1e-2', '1e-1', '1e-0'], fontsize=fontsize)

leg = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5, ncol=2)

for legobj in leg.legendHandles:
    legobj.set_linewidth(3.)

plt.savefig(
    output_path.joinpath(sim02_setup['name']+'_tron_zoomed.png'),
    dpi=300,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)

# %% Generate and save zoomed in SGD figure for simulation 2

plt.figure(figsize=[8, 5])

xrange = range(1, sgd_error_loaded.shape[0]+1)

labels = [r'$\theta_\ast$ = {}'.format(sim02_setup['boundlist'][i]) for i in range(sgd_error_loaded.shape[1])]

for i in range(sgd_error_loaded.shape[1]):
    plt.plot(
        xrange,
        np.log10(sgd_error_loaded[:, i]),
        linewidth=2.,
        label=labels[i]
    )

plt.ylim([-4.2, 0.2])   

plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

xtickstep = 5000
xticks = range(0, sgd_error_loaded.shape[0]+xtickstep, xtickstep)
plt.xticks(ticks=xticks, fontsize=fontsize)

yticks = [-4, -3, -2, -1, 0]
plt.yticks(ticks=yticks, labels=['1e-4', '1e-3', '1e-2', '1e-1', '1e-0'], fontsize=fontsize)

leg = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5, ncol=2)

for legobj in leg.legendHandles:
    legobj.set_linewidth(3.)

plt.savefig(
    output_path.joinpath(sim02_setup['name']+'_sgd_zoomed.png'),
    dpi=300,
    pil_kwargs={'quality': 100},
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1
)



# %% Generate and save zoomed in NeuroTron-vs-SGD figure for simulation 2

for i in range(1, tron_error_loaded.shape[1]):
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

    plt.ylim([-4.2, 1.2])   

    plt.title(r'Laplace data, $\theta_\ast$ = {}'.format(sim02_setup['boundlist'][i]))

    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(r'Parameter error ($\log_{10}$ scale)', fontsize=fontsize)

    xtickstep = 5000
    xticks = range(0, sgd_error_loaded.shape[0]+xtickstep, xtickstep)
    plt.xticks(ticks=xticks, fontsize=fontsize)

    yticks = [-4, -3, -2, -1, 0, 1]
    plt.yticks(ticks=yticks, labels=['1e-4', '1e-3', '1e-2', '1e-1', '1e-0', '1e+1'], fontsize=fontsize)

    leg = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.5, ncol=2)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.)

    plt.savefig(
        output_path.joinpath(
            sim02_setup['name']+'_tron_vs_sgd_zoomed_theta_val'+str(i).zfill(len(str(tron_error_loaded.shape[1])))+'.png'
        ),
        dpi=300,
        pil_kwargs={'quality': 100},
        transparent=True,
        bbox_inches='tight',
        pad_inches=0.1
    )
