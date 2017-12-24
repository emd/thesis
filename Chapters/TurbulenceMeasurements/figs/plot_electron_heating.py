import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct


# Shots analyzed
shots = np.array([
    171536,
    171538
])

# Corresponding ECH locations
rho_ECH = np.array([
    0.5,
    0.8
])

# Corresponding analysis times
# [times] = ms
times = np.array([
    2750,
    2200
])

# Plotting parameters
figsize = (8, 4)
cols = get_distinct(2)
linewidth = 2
fontsize = 16


def overview_plot():
    qs = np.array([
        'qbeam_e',
        'qohm_e',
        'qrad_e',
        'qrf_e'
    ])

    linestyles = ['-', '--', '-.', ':']

    plt.figure()

    for sind, shot in enumerate(shots):
        dname = './%i/%ibis' % (shot, times[sind])

        fname = '%s/%s.pkl' % (dname, 'rhon_grid')
        with open(fname, 'rb') as f:
            rho = pickle.load(f)

        for qind, q in enumerate(qs):
            fname = '%s/%s.pkl' % (dname, q)
            with open(fname, 'rb') as f:
                q = pickle.load(f)

            plt.plot(
                rho,
                q / 1e3,
                color=cols[sind],
                linestyle=linestyles[qind],
                linewidth=linewidth)

    plt.xlabel(r'$\mathregular{\rho}$', fontsize=fontsize)
    plt.ylabel(r'$\mathregular{q \; [kW / m^3]}$', fontsize=fontsize)
    plt.show()

    return


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=figsize)

    for sind, shot in enumerate(shots):
        dname = './%i/%ibis' % (shot, times[sind])

        fname = '%s/%s.pkl' % (dname, 'rhon_grid')
        with open(fname, 'rb') as f:
            rho = pickle.load(f)

        fname = '%s/%s.pkl' % (dname, 'qrf_e')
        with open(fname, 'rb') as f:
            qrf_e = pickle.load(f)

        fname = '%s/%s.pkl' % (dname, 'qbeam_e')
        with open(fname, 'rb') as f:
            qbeam_e = pickle.load(f)

        axs[0].plot(
            rho,
            qrf_e / 1e6,
            color=cols[sind],
            linewidth=linewidth)

        axs[1].plot(
            rho,
            qbeam_e / 1e3,
            color=cols[sind],
            linewidth=linewidth,
            label=(r'$\mathregular{\rho_{ECH} = %.1f}$' % rho_ECH[sind]))

    axs[0].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[0].set_ylabel(
        r'$\mathregular{q_{e,ECH} \; [MW / m^3]}$',
        fontsize=fontsize)

    axs[1].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[1].set_ylabel(
        r'$\mathregular{q_{e,NBI} \; [kW / m^3]}$',
        fontsize=fontsize)
    axs[1].legend(
        loc='upper right',
        fontsize=(fontsize - 2))

    plt.tight_layout()
    plt.show()
