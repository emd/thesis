import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct
from plot_Er_profiles import get_Er_midplane, get_gammaE


# Shots analyzed
shots = np.array([
    171536,
    171538
])

# Corresponding analysis times
# [times] = ms
times = np.array([
    2750,
    2200
])

# Profiles to plot
# (Note that Er will also be plotted, but this data
# is on a different grid etc., so it is not straightforward
# to simply loop through the Er profile...)
profiles = np.array([
    'ne',
    # 'ni1',  # thermal D density
    'te',
    'ti1'   # thermal D temperature
])

# Plotting parameters
rholim = [0.0, 1.0]
aLy_lim = [-1, 4]
figsize = (9, 9)
figsize = (8, 8)
linewidth = 2
fontsize = 15
Nsmooth_gammaE = 5
gammaE_lim = [-10, 50]


if __name__ == '__main__':
    normalizations = {
        'ne': 1e13,
        'te': 1.,
        'ni1': 1e13,
        'ti1': 1.
    }

    profile_ylabels = {
        'ne': r'$\mathregular{n_e \; [10^{19} \; m^{-3}]}$',
        'te': r'$\mathregular{T_e \; [keV]}$',
        'ni1': r'$\mathregular{n_i \; [10^{19} \; m^{-3}]}$',
        'ti1': r'$\mathregular{T_i \; [keV]}$'
    }

    scalelength_ylabels= {
        'ne': r'$\mathregular{a \, / \, L_{n_e}}$',
        'te': r'$\mathregular{a \, / \, L_{T_e}}$',
        'ni1': r'$\mathregular{a \, / \, L_{n_i}}$',
        'ti1': r'$\mathregular{a \, / \, L_{T_i}}$'
    }

    fig, ax = plt.subplots(
        len(profiles) + 1,  # 1 extra row for Er
        2,
        sharex=True,
        figsize=figsize)
    cols = get_distinct(len(shots))

    for sind, shot in enumerate(shots):
        # Load spatial coordinate, rho
        fname = './%i/%i/rho.pkl' % (shot, times[sind])
        with open(fname, 'rb') as f:
            rho = pickle.load(f)[0, :]

        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]

        # Load each profile and plot
        for pind, profile in enumerate(profiles):
            # Load profile
            fname = './%i/%i/%s.pkl' % (shot, times[sind], profile)
            with open(fname, 'rb') as f:
                y = pickle.load(f)[0, :]

            # Load corresponding scale length
            fname = './%i/%i/aL%s.pkl' % (shot, times[sind], profile)
            with open(fname, 'rb') as f:
                aLy = pickle.load(f)[0, :]

            # Plot
            ax[pind, 0].plot(
                rho[rhoind],
                y[rhoind] / normalizations[profile],
                c=cols[sind],
                linewidth=linewidth)
            ax[pind, 0].set_ylabel(
                profile_ylabels[profile],
                fontsize=fontsize)

            ax[pind, 1].plot(
                rho[rhoind],
                aLy[rhoind],
                c=cols[sind],
                linewidth=linewidth)
            ax[pind, 1].set_ylabel(
                scalelength_ylabels[profile],
                fontsize=fontsize)
            ax[pind, 1].set_ylim(aLy_lim)

        # Er profile
        rho, Er_midplane = get_Er_midplane(
            shot,
            times[sind])
        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]
        ax[-1, 0].plot(
            rho[rhoind],
            Er_midplane[rhoind],
            color=cols[sind],
            linewidth=linewidth)
        ax[-1, 0].set_ylabel(
            r'$\mathregular{E_r \; [kV / m]}$',
            fontsize=fontsize)

        # gammaE
        rho, gammaE = get_gammaE(
            shot,
            times[sind],
            Nsmooth=Nsmooth_gammaE)
        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]
        ax[-1, 1].plot(
            rho[rhoind],
            gammaE[rhoind] / 1e3,
            color=cols[sind],
            linewidth=linewidth)
        ax[-1, 1].set_ylabel(
            r'$\mathregular{\gamma_E \; [kHz]}$',
            fontsize=fontsize)
        ax[-1, 1].set_ylim(gammaE_lim)

    xlabel = r'$\mathregular{\rho}$'
    ax[-1, 0].set_xlabel(xlabel, fontsize=fontsize)
    ax[-1, 1].set_xlabel(xlabel, fontsize=fontsize)
    ax[-1, 0].set_xlim(rholim)
    ax[-1, 1].set_xlim(rholim)

    legend = [
        '%i, %i ms' % (shots[0], times[0]),
        '%i, %i ms' % (shots[1], times[1]),
    ]
    ax[0, 1].legend(
        legend,
        fontsize=(fontsize - 2),
        loc='upper left')

    ax[1, 0].set_yticks(np.arange(0, 5, 1))
    ax[2, 0].set_yticks(np.arange(0, 5, 1))

    # Subplot labels
    x0 = 0.02
    ax[0, 0].annotate('(a)', (x0, 1.25), fontsize=fontsize)
    ax[1, 0].annotate('(b)', (x0, 0.2), fontsize=fontsize)
    ax[2, 0].annotate('(c)', (x0, 0.2), fontsize=fontsize)
    ax[3, 0].annotate('(d)', (x0, -37), fontsize=fontsize)
    ax[0, 1].annotate('(e)', (x0, -0.75), fontsize=fontsize)
    ax[1, 1].annotate('(f)', (x0, -0.75), fontsize=fontsize)
    ax[2, 1].annotate('(g)', (x0, -0.75), fontsize=fontsize)
    ax[3, 1].annotate('(h)', (x0, 42), fontsize=fontsize)

    plt.tight_layout()
    plt.show()
