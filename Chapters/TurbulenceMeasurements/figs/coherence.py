import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from distinct_colours import get_distinct

import random_data as rd
import mitpci


shots = [
    171536,
    171538
]
rhos = [
    0.5,
    0.8
]
# tlims = [
#     [2.5, 3.2],
#     [2.0, 2.8]
# ]
tlims = [
    [2.65, 2.85],
    [2.10, 2.30]
]
pci_channel = 8

# Conversion factors
Hz_per_kHz = 1e3

# ELM-filtering parameters
sigma_mult = 3
debounce_dt = 0.5e-3            # Less than ELM spacing
window_fraction = [0.2, 0.8]

# Spectral parameters
Tens = 0.5e-3       # Less than ELM spacing
Nreal_per_ens = 5

# Plotting parameters
cols = get_distinct(2)
linewidth = 2
fontsize = 18
mpl.rcParams['xtick.labelsize'] = fontsize - 3
mpl.rcParams['ytick.labelsize'] = fontsize - 3


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)

    for sind, shot in enumerate(shots):
        rho = rhos[sind]
        tlim = tlims[sind]

        # Load data
        L = mitpci.interferometer.Lissajous(shot, tlim=tlim)
        Ph_int = mitpci.interferometer.Phase(L)
        Ph_pci = mitpci.pci.Phase(shot, pci_channel, tlim=tlim)

        # Get FFTs, removing last dimension corresponding
        # to realizations per ensemble
        ens = rd.ensemble.Ensemble(
            Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
            Tens=Tens, Nreal_per_ens=1)
        Xk = np.squeeze(ens.getFFTs(Ph_int.x))
        Yk = np.squeeze(ens.getFFTs(Ph_pci.x))

        # ELM filter
        SH = rd.signals.SpikeHandler(
            Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
            sigma_mult=sigma_mult, debounce_dt=debounce_dt)
        tind = SH.getSpikeFreeTimeIndices(
            ens.t, window_fraction=window_fraction)
        Nreal = len(tind)
        print '\nNreal: %i' % Nreal

        # Compute ELM-filtered coherence
        Gxy = np.mean(np.conj(Xk[:, tind]) * Yk[:, tind], axis=-1)
        Gxx = np.mean((np.abs(Xk[:, tind])) ** 2, axis=-1)
        Gyy = np.mean((np.abs(Yk[:, tind])) ** 2, axis=-1)
        gamma2xy = (np.abs(Gxy)) ** 2 / (Gxx * Gyy)

        # Unit conversions
        f = ens.f / Hz_per_kHz
        find = np.where(np.logical_and(
            f >= 10,
            f <= 1e3))[0]

        # Plot
        ax.plot(
            f[find],
            gamma2xy[find],
            c=cols[sind],
            linewidth=linewidth)

    # Labeling
    ax.set_ylabel(
        r'$\mathregular{\gamma^2_{xy}}$',
        fontsize=fontsize)
    ax.set_xlabel(
        r'$\mathregular{f \; [kHz]}$',
        fontsize=fontsize)
    labels = [
        r'$\mathregular{\rho_{ECH} = 0.5}$',
        r'$\mathregular{\rho_{ECH} = 0.8}$'
    ]
    ax.legend(
        labels,
        loc='upper right',
        fontsize=(fontsize - 2))

    # # Add shot numbers & times
    # x0 = 1.6e2

    # sind = 0
    # ax.annotate(
    #     '%i, [%.2f, %.2f] s' % (shots[sind], tlims[sind][0], tlims[sind][1]),
    #     (x0, 4.4e-8),
    #     color=cols[sind],
    #     fontsize=(fontsize - 6))
    # sind = 1
    # ax.annotate(
    #     '%i, [%.2f, %.2f] s' % (shots[sind], tlims[sind][0], tlims[sind][1]),
    #     (x0, 2.4e-8),
    #     color=cols[sind],
    #     fontsize=(fontsize - 6))

    # # Limits and tick marks
    # plt.xlim([10, 1000])
    # ax.xaxis.set_major_formatter(
    #     mpl.ticker.FormatStrFormatter('%d'))
    # ax.set_xticks([10, 100, 1000])

    plt.show()
