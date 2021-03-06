import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from linear_stability import load_data as load_data_single_rho
import normalization


shots = [171536, 171538]
times = [2750, 2200]
drho = 0.05
rhos = np.arange(0.3, 0.95, drho)

# Plotting parameters
figsize = (8, 6.5)
fontsize = 15
gamma_lim = [1, 2e3]   # [gamma_lim] = kHz
vph_lim = [-1.8, 1.8]  # [vph_lim] = km / s


def load_data(shot, time, rhos, physical_units=True):
    data = load_data_single_rho(shot, time, rhos[0])

    Nky = len(data['ky'])

    ky = np.zeros((Nky, len(rhos)))
    gamma = np.zeros((Nky, len(rhos)))
    omega = np.zeros((Nky, len(rhos)))

    for ind, rho in enumerate(rhos):
        data = load_data_single_rho(shot, time, rho)

        # Parse data
        ky[:, ind] = data['ky']         # [ky] = 1 / rho_s
        gamma[:, ind] = data['gamma1']  # [gamma] = c_s / a
        omega[:, ind] = data['freq1']   # [omega] = c_s / a

    vph = omega / ky                    # [vph] = c_s * (rho_s / a)

    if physical_units:
        # Load normalizations
        dmp, c_s = normalization.get_c_s(
            shot, time, rhointerp=rhos)
        dmp, rho_s_a = normalization.get_rho_s_a(
            shot, time, rhointerp=rhos)
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
    fig, axs = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=figsize)

    for sind, shot in enumerate(shots):
        # Load data
        time = times[sind]
        data = load_data(shot, time, rhos, physical_units=True)

        # Parse data
        ky = data['ky']        # [ky] = 1 / rho_s
        gamma = data['gamma']  # [gamma] = kHz
        vph = data['vph']      # [vph] = km / s

        # Plots
        kind = 0

        gamma_cmap = plt.get_cmap('viridis')
        gamma_cmap.set_bad('gray')

        # See pcolormesh description here:
        #
        #   https://stackoverflow.com/a/43129331/5469497
        #
        # to understand the grid. Also look at documentation
        # for `plt.pcolor`. More important to get this right
        # for coarsely spaced rho than for more finely space ky;
        # also, its easier for the uniformly spaced rho grid
        # than for the non-uniformly spaced ky grid.
        rhogrid = np.arange(
            rhos[0] - (0.5 * drho),
            rhos[-1] + (1.5 * drho),
            drho)

        m = axs[0, sind].pcolormesh(
            rhogrid,
            ky[:, kind],
            gamma,
            vmin=gamma_lim[0],
            vmax=gamma_lim[1],
            norm=LogNorm(vmin=gamma_lim[0], vmax=gamma_lim[1]),
            cmap=gamma_cmap)
        cb = plt.colorbar(m, ax=axs[0, sind] , extend='min')
        m.cmap.set_under('gray')
        m.set_edgecolor('face')  # avoid grid lines in PDF file
        cb.set_label(
            r'$\mathregular{\gamma \; [kHz]}$',
            fontsize=fontsize)

        m = axs[1, sind].pcolormesh(
            rhogrid,
            ky[:, kind],
            vph,
            cmap='BrBG',
            vmin=vph_lim[0],
            vmax=vph_lim[1])
        cb = plt.colorbar(m, ax=axs[1, sind])
        m.set_edgecolor('face')  # avoid grid lines in PDF file
        cb.set_label(
            r'$\mathregular{v_{ph} \; [km / s]}$',
            fontsize=fontsize)

    # Plot limits and scale
    axs[0, 0].set_xlim([rhogrid[0], rhogrid[-1]])
    axs[0, 0].set_ylim([ky[0, kind], ky[-1, kind]])
    axs[0, 0].set_yscale('log')

    # Labeling
    axs[1, 0].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[1, 1].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[0, 0].set_ylabel(
        r'$\mathregular{k_y \rho_s}$',
        fontsize=fontsize)
    axs[1, 0].set_ylabel(
        r'$\mathregular{k_y \rho_s}$',
        fontsize=fontsize)
    axs[0, 0].set_title(
        r'$\mathregular{\rho_{ECH} = 0.5}$',
        fontsize=fontsize)
    axs[0, 1].set_title(
        r'$\mathregular{\rho_{ECH} = 0.8}$',
        fontsize=fontsize)

    plt.tight_layout()
    plt.show()
