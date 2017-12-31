import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct

from linear_stability import load_data as load_data_single_rho
import normalization


shots = [171536, 171538]
# shots = [171536]
times = [2750, 2200]
rhos = np.arange(0.3, 0.95, 0.05)
# rhos = np.arange(0.3, 0.8, 0.05)


# Plotting parameters
figsize = (8, 6.5)
fontsize = 15
cols = get_distinct(2)
gamma_lim = [1, 2e3]   # [gamma_lim] = kHz
vph_lim = [-1.8, 1.8]  # [vph_lim] = km / s


# Helper class for creating a diverging colormap with asymmetric limits;
# taken from Joe Kington at:
#
#   https://stackoverflow.com/a/20146989/5469497
#
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def load_data(shot, time, rhos):
    data = load_data_single_rho(shot, time, rhos[0])

    Nky = len(data['ky'])

    ky = np.zeros((Nky, len(rhos)))
    gamma1 = np.zeros((Nky, len(rhos)))
    freq1 = np.zeros((Nky, len(rhos)))

    for ind, rho in enumerate(rhos):
        data = load_data_single_rho(shot, time, rho)

        freq1[:, ind] = data['freq1']
        gamma1[:, ind] = data['gamma1']
        ky[:, ind] = data['ky']

    # Package data
    data = {
        'freq1': freq1,
        'gamma1': gamma1,
        'ky': ky
    }

    return data


if __name__ == '__main__':
    fig, axs = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=figsize)

    for sind, shot in enumerate(shots):
        # Load data
        time = times[sind]
        data = load_data(shot, time, rhos)

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

        # Parse data
        ky = data['ky']         # [ky] = 1 / rho_s
        gamma = data['gamma1']  # [gamma] = c_s / a
        omega = data['freq1']   # [omega] = c_s / a
        vph = omega / ky        # [vph] = c_s * (rho_s / a)

        # Convert to physical units
        gamma *= (c_s / a)      # [gamma] = kHz
        vph *= (c_s * rho_s_a)  # [vph] = km / s

        # Mask stable modes
        gamma = np.ma.masked_where(
            gamma <= 0,
            gamma)

        # Find maximum and (non-zero) minimum of growth rate
        gamma_min = np.ma.min(gamma)
        gamma_max = np.ma.max(gamma)

        # Find maximum absolute phase velocity
        vphmax = np.max(np.abs(vph))

        # Plots
        kind = 0

        m = axs[0, sind].pcolormesh(
            rhos,
            ky[:, kind],
            gamma,
            vmin=gamma_lim[0],
            vmax=gamma_lim[1],
            norm=LogNorm(),
            cmap='viridis')
        cb = plt.colorbar(m, ax=axs[0, sind] , extend='min')
        m.cmap.set_under('gray')
        m.set_edgecolor('face')  # avoid grid lines in PDF file
        cb.set_label(
            r'$\mathregular{\gamma \; [kHz]}$',
            fontsize=fontsize)

        m = axs[1, sind].pcolormesh(
            rhos,
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
    axs[0, 0].set_xlim([rhos[0], rhos[-1]])
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
