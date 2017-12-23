import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct
from plot_Er_profiles import get_Er_midplane, get_gammaE
from get_uncertainty import Uncertainty, Measurements
from plot_Er_uncertainty_force_balance import (
    get_Er_uncertainty_force_balance,
    get_dEr_uncertainty_force_balance)


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

rhomin_PCI = 0.35

# Plotting parameters
rholim = [0.0, 1.0]
aLy_lim = [-1, 4]
figsize = (9, 9)
figsize = (8, 8)
linewidth = 1
alpha = 0.5
fontsize = 15
Er_lim = [-10, 20]
Nsmooth_gammaE = 10
rholim_gammaE = [0.1, 1.0]
gammaE_lim = [0, 50]
fillcolor = 'lightgray'
plot_measurements = True
measurements_shot = shots[0]
marker = 'o'
markersize = 3


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

    ymax = {
        'ne': 7,
        'te': 4,
        'ti1': 4
    }

    fig, ax = plt.subplots(
        len(profiles) + 1,  # 1 extra row for Er
        2,
        sharex=True,
        figsize=figsize)
    cols = get_distinct(len(shots))

    for sind, shot in enumerate(shots):
        # Load spatial coordinate, rho
        fname = './%i/%ibis/rho.pkl' % (shot, times[sind])
        with open(fname, 'rb') as f:
            rho = pickle.load(f)[0, :]

        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]

        # Load each profile and plot
        for pind, profile in enumerate(profiles):
            # Load profile
            fname = './%i/%ibis/%s.pkl' % (shot, times[sind], profile)
            with open(fname, 'rb') as f:
                y = pickle.load(f)[0, :]

            # Load corresponding scale length
            fname = './%i/%ibis/aL%s.pkl' % (shot, times[sind], profile)
            with open(fname, 'rb') as f:
                aLy = pickle.load(f)[0, :]

            # Get corresponding uncertainties
            U = Uncertainty(profile, shot, rho=rho)

            y /= normalizations[profile]

            # Load measurement points, if requested
            if plot_measurements and (shot == measurements_shot):
                M = Measurements(profile, shot)

            # Plot

            # First, annotate region inaccessible by PCI
            if sind == 0:
                ax[pind, 0].fill_betweenx(
                    [0, ymax[profile]],
                    0,
                    rhomin_PCI,
                    color=fillcolor)
                ax[pind, 1].fill_betweenx(
                    aLy_lim,
                    0,
                    rhomin_PCI,
                    color=fillcolor)
                if pind == 0:
                    ax[pind, 0].annotate(
                        r'$\mathregular{R < 1.98 \; m}$',
                        (0.02, 3.1),
                        fontsize=(fontsize - 3))

            # Plot measurement points & error bars, if requested
            if plot_measurements and (shot == measurements_shot):
                M.plot(
                    rho_lim=rholim,
                    ax=ax[pind, 0],
                    color=cols[sind],
                    marker=marker,
                    markersize=markersize)

            ax[pind, 0].plot(
                rho[rhoind],
                y[rhoind],
                c=cols[sind],
                linewidth=linewidth)
            ax[pind, 0].fill_between(
                rho[rhoind],
                ((1 - U.y_relerr) * y)[rhoind],
                ((1 + U.y_relerr) * y)[rhoind],
                color=cols[sind],
                alpha=alpha)
            ax[pind, 0].set_ylabel(
                profile_ylabels[profile],
                fontsize=fontsize)
            ax[pind, 0].set_ylim([0, ymax[profile]])

            ax[pind, 1].plot(
                rho[rhoind],
                aLy[rhoind],
                c=cols[sind],
                linewidth=linewidth)
            ax[pind, 1].fill_between(
                rho[rhoind],
                ((1 - U.aLy_relerr) * aLy)[rhoind],
                ((1 + U.aLy_relerr) * aLy)[rhoind],
                color=cols[sind],
                alpha=alpha)
            ax[pind, 1].set_ylabel(
                scalelength_ylabels[profile],
                fontsize=fontsize)
            ax[pind, 1].set_ylim(aLy_lim)

        # Er profile
        if sind == 0:
            ax[-1, 0].fill_betweenx(
                Er_lim,
                0,
                rhomin_PCI,
                color=fillcolor)

        rho, Er_midplane = get_Er_midplane(
            shot,
            times[sind])
        rho, Er_relerr = get_Er_uncertainty_force_balance(
            shot,
            rhointerp=rho)
        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]

        ax[-1, 0].plot(
            rho[rhoind],
            Er_midplane[rhoind],
            color=cols[sind],
            linewidth=linewidth)

        # Enforce 8 kV/m maximum error...
        # kinda jenky, but only place this comes into play
        # is in rho > 0.9, where we don't care...
        delta_Er = Er_relerr * Er_midplane
        delta_Er = np.minimum(delta_Er, 8)
        delta_Er = np.maximum(delta_Er, -8)

        ax[-1, 0].fill_between(
            rho[rhoind],
            (Er_midplane - delta_Er)[rhoind],
            (Er_midplane + delta_Er)[rhoind],
            color=cols[sind],
            alpha=alpha)
        ax[-1, 0].set_ylabel(
            r'$\mathregular{E_r \; [kV / m]}$',
            fontsize=fontsize)
        ax[-1, 0].set_ylim(Er_lim)

        # gammaE
        if sind == 0:
            ax[-1, 1].fill_betweenx(
                gammaE_lim,
                0,
                rhomin_PCI,
                color=fillcolor)

        rho, gammaE = get_gammaE(
            shot,
            times[sind],
            Nsmooth=Nsmooth_gammaE)
        rho, dEr_relerr = get_dEr_uncertainty_force_balance(
            shot,
            rhointerp=rho,
            Nsmooth=Nsmooth_gammaE)
        rhoind = np.where(np.logical_and(
            rho >= rholim_gammaE[0],
            rho <= rholim_gammaE[1]))[0]

        gammaE /= 1e3

        ax[-1, 1].plot(
            rho[rhoind],
            gammaE[rhoind],
            color=cols[sind],
            linewidth=linewidth)

        # Enforce 20 kHz maximum error...
        # kinda jenky, but only place this comes into play
        # rho > 0.9, where we don't care (off of plot range)
        delta_gammaE = dEr_relerr * gammaE
        delta_gammaE = np.minimum(delta_gammaE, 20)
        delta_gammaE = np.maximum(delta_gammaE, -20)
        delta_gammaE = np.convolve(
            delta_gammaE,
            np.ones(Nsmooth_gammaE, dtype='float') / Nsmooth_gammaE,
            mode='same')

        ax[-1, 1].fill_between(
            rho[rhoind],
            (gammaE - delta_gammaE)[rhoind],
            (gammaE + delta_gammaE)[rhoind],
            color=cols[sind],
            alpha=alpha)
        ax[-1, 1].set_ylabel(
            r'$\mathregular{\gamma_E \; [kHz]}$',
            fontsize=fontsize)
        ax[-1, 1].set_ylim(gammaE_lim)

    xlabel = r'$\mathregular{\rho}$'
    ax[-1, 0].set_xlabel(xlabel, fontsize=fontsize)
    ax[-1, 1].set_xlabel(xlabel, fontsize=fontsize)
    ax[-1, 0].set_xlim(rholim)
    ax[-1, 1].set_xlim(rholim)

    legend_labels = [
        r'$\mathregular{\rho_{ECH} = %.1f}$' % rho_ECH[0],
        r'$\mathregular{\rho_{ECH} = %.1f}$' % rho_ECH[1]
    ]
    leg = ax[0, 1].legend(
        legend_labels,
        fontsize=(fontsize - 2),
        loc='upper left')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)

    ax[1, 0].set_yticks(np.arange(0, 5, 1))
    ax[2, 0].set_yticks(np.arange(0, 5, 1))

    # Subplot labels
    x0 = 0.02
    ax[0, 0].annotate('(a)', (x0, 0.35), fontsize=fontsize)
    ax[1, 0].annotate('(b)', (x0, 0.2), fontsize=fontsize)
    ax[2, 0].annotate('(c)', (x0, 0.2), fontsize=fontsize)
    ax[3, 0].annotate('(d)', (x0, -8.5), fontsize=fontsize)
    ax[0, 1].annotate('(e)', (x0, -0.75), fontsize=fontsize)
    ax[1, 1].annotate('(f)', (x0, -0.75), fontsize=fontsize)
    ax[2, 1].annotate('(g)', (x0, -0.75), fontsize=fontsize)
    ax[3, 1].annotate('(h)', (x0, 2), fontsize=fontsize)

    # Shot and time annotations
    for sind, shot in enumerate(shots):
        tmid = times[sind] * 1e-3
        dt = 0.1
        t0 = tmid - dt
        tf = tmid + dt

        x0 = 0.51
        y0 = -0.35
        dy = 0.5

        ax[0, 1].annotate(
            '%i, [%.2f, %.2f] s' % (shot, t0, tf),
            (x0, y0 - (sind * dy)),
            color=cols[sind],
            fontsize=(fontsize - 6))

    plt.tight_layout()
    plt.show()
