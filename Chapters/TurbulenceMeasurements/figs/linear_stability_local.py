import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct

import normalization


shots = [171536, 171538]
times = [2750, 2200]
rho_ECH = [0.5, 0.8]
rho = 0.6

# Plotting parameters
fontsize = 15
linestyle='-'
linewidth = 2
markers = ['o', 's']
cols = get_distinct(2)


def load_data(shot, time, rho, physical_units=True):
    d = '%i/%ibis/0%i' % (shot, time, np.int(np.round(100 * rho)))

    # Load data
    fname = '%s/%s.pkl' % (d, 'ky')
    with open(fname, 'rb') as f:
        ky = pickle.load(f)         # [ky] = 1 / rho_s

    fname = '%s/%s.pkl' % (d, 'freq1')
    with open(fname, 'rb') as f:
        omega = pickle.load(f)      # [omega] = c_s / a

    fname = '%s/%s.pkl' % (d, 'gamma1')
    with open(fname, 'rb') as f:
        gamma = pickle.load(f)      # [gamma] = c_s / a

    vph = omega / ky                # [vph] = c_s * (rho_s / a)

    if physical_units:
        # Load normalizations
        dmp, c_s = normalization.get_c_s(
            shot, time, rhointerp=rho)
        dmp, rho_s_a = normalization.get_rho_s_a(
            shot, time, rhointerp=rho)
        a = normalization.get_a(
            shot, time)

        # Convert sound speed to km/s
        c_s *= 1e-3

        # Convert a to m
        a *= 1e-2

        # Convert to physical units
        gamma *= (c_s / a)      # [gamma] = kHz
        omega *= (c_s / a)      # [omega] = kHz
        vph *= (c_s * rho_s_a)  # [vph] = km / s

    # Package data
    data = {
        'ky': ky,
        'gamma': gamma,
        'omega': omega,
        'vph': vph,
    }

    return data


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 1, sharex=True)

    for sind, shot in enumerate(shots):
        # Load data
        data = load_data(
            shot,
            times[sind],
            rho,
            physical_units=True)

        # Parse data
        ky = data['ky']
        gamma = data['gamma']
        omega = data['omega']
        vph = data['vph']

        axs[0].loglog(
            ky,
            gamma,
            color=cols[sind],
            marker=markers[sind],
            linewidth=linewidth,
            linestyle=linestyle)
        axs[1].semilogx(
            ky,
            vph,
            color=cols[sind],
            marker=markers[sind],
            linewidth=linewidth,
            linestyle=linestyle,
            label=r'$\mathregular{\rho_{ECH} = %.1f}$' % rho_ECH[sind])

    # Labeling
    axs[1].set_xlabel(
        r'$\mathregular{k_y \rho_s}$',
        fontsize=fontsize)
    axs[0].set_ylabel(
        r'$\mathregular{\gamma \; [kHz]}$',
        fontsize=fontsize)
    axs[1].set_ylabel(
        r'$\mathregular{v_{ph} \; [km / s]}$',
        fontsize=fontsize)
    axs[1].legend(
        loc='best',
        fontsize=(fontsize - 1))

    # Add zero line
    axs[1].axhline(
        0,
        xmin=1e-2,  # Don't know why using ky[0] is misbehaving?
        xmax=ky[-1],
        linestyle='--',
        linewidth=linewidth,
        color='k')

    # Set limits
    xlim = [ky[0], ky[-1]]
    ylim0 = [3, 1e3]
    ylim1 = [-1, 1]

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim0)
    axs[1].set_ylim(ylim1)

    # Gray out and annotate region beyond PCI-interferometer range
    kmax = 4
    fillcolor = 'gray'
    alpha = 0.5

    axs[0].fill_betweenx(
        ylim0,
        kmax,
        xlim[-1],
        color=fillcolor,
        alpha=alpha)
    axs[1].fill_betweenx(
        ylim1,
        kmax,
        xlim[-1],
        color=fillcolor,
        alpha=alpha)
    axs[0].annotate(
        r'$\mathregular{k_R > 25 \, cm^{-1}}$',
        (6, 30),
        fontsize=(fontsize - 1))

    # Annotate radial coordinate
    axs[0].annotate(
        r'$\mathregular{\rho = %.2f}$' % rho,
        (1.075 * xlim[0], 0.5 * ylim0[1]),
        color='k',
        fontsize=(fontsize - 1))

    plt.show()
