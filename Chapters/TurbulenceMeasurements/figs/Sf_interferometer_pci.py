import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random_data as rd
import mitpci
from gadata import gadata
from distinct_colours import get_distinct

from noise_floor import (
    interferometer_noise_floor_model,
    pci_noise_floor_model)


shots = [
    171536,
    171538
]
rhos = [
    0.5,
    0.8
]
tlims = [
    [2.5, 3.2],
    [2.0, 2.8]
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
Nreal_per_ens = 1   # Do averaging after ELM-filtering

# Plotting parameters
figsize = (10, 5)
cols = get_distinct(2)
linewidth = 2
fontsize = 18
mpl.rcParams['xtick.labelsize'] = fontsize - 3
mpl.rcParams['ytick.labelsize'] = fontsize - 3

# Remove interferometer coherent pickup
fcoherent = [
    [321993, 345681],
    [490328, 514368],
    [654570, 682260],
    [825666, 850316],
    [990271, 1017980]
]


if __name__ == '__main__':
    fig, axes = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=figsize)

    for sind, shot in enumerate(shots):
        rho = rhos[sind]
        tlim = tlims[sind]

        # Load data
        L = mitpci.interferometer.Lissajous(shot, tlim=tlim)
        Ph_int = mitpci.interferometer.Phase(L)
        Ph_pci = mitpci.pci.Phase(shot, pci_channel, tlim=tlim)

        # Compute spectra
        asd_int = rd.spectra.AutoSpectralDensity(
            Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
            Tens=Tens, Nreal_per_ens=Nreal_per_ens)
        asd_pci = rd.spectra.AutoSpectralDensity(
            Ph_pci.x, Fs=Ph_pci.Fs, t0=Ph_pci.t0,
            Tens=Tens, Nreal_per_ens=Nreal_per_ens)

        # ELM filter
        SH = rd.signals.SpikeHandler(
            Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
            sigma_mult=sigma_mult, debounce_dt=debounce_dt)

        # Remove coherent peaks from interferometer spectra
        for fc in fcoherent:
            find = np.where(np.logical_and(
                asd_int.f >= fc[0],
                asd_int.f <= fc[1]))[0]

            asd_int.Gxx[find, :] = 0
            asd_int.Gxx[find, :] = np.mean([
                asd_int.Gxx[find[0] - 1, :],
                asd_int.Gxx[find[-1] + 1, :]],
                axis=0)

        find_int = np.where(np.logical_and(
            asd_int.f >= 10e3,
            asd_int.f <= 1e6))[0]
        find_pci = np.where(np.logical_and(
            asd_pci.f >= 10e3,
            asd_pci.f <= 2e6))[0]

        # Unit conversions
        asd_int.f /= Hz_per_kHz
        asd_pci.f /= Hz_per_kHz
        asd_int.Gxx *= Hz_per_kHz
        asd_pci.Gxx *= Hz_per_kHz

        # Plot spectra
        ind = SH.getSpikeFreeTimeIndices(
            asd_int.t, window_fraction=window_fraction)
        axes[0].loglog(
            asd_int.f[find_int],
            np.mean(asd_int.Gxx[find_int, :][:, ind], axis=-1),
            c=cols[sind],
            linewidth=linewidth)
        axes[1].loglog(
            asd_pci.f[find_pci],
            np.mean(asd_pci.Gxx[find_pci, :][:, ind], axis=-1),
            c=cols[sind],
            linewidth=linewidth)

    # Plot noise floors
    axes[0].loglog(
        asd_int.f[find_int],
        interferometer_noise_floor_model(asd_int.f[find_int]),
        linestyle='--',
        linewidth=linewidth,
        c='k')
    axes[1].loglog(
        asd_pci.f[find_pci],
        pci_noise_floor_model(asd_pci.f[find_pci]),
        linestyle='--',
        linewidth=linewidth,
        c='k')
    axes[0].annotate(
        'noise floor',
        xy=(12, 1.2e-10),
        fontsize=(fontsize - 2))
    axes[1].annotate(
        'noise floor',
        xy=(12, 1.2e-12),
        fontsize=(fontsize - 2))

    # Labeling
    axes[0].set_title(
        r'$\mathregular{interferometer \; (|k_R| < 5 \; cm^{-1})}$',
        fontsize=(fontsize - 1))
    axes[1].set_title(
        r'$\mathregular{PCI \; (1.5 \; cm^{-1} < |k_R| \leq 25 \; cm^{-1})}$',
        fontsize=(fontsize - 1))
    axes[0].set_ylabel(
        r'$\mathregular{G_{\phi,\phi}(f) \; [rad^2 /\, kHz]}$',
        fontsize=fontsize)
    axes[0].set_xlabel(
        r'$\mathregular{f \; [kHz]}$',
        fontsize=fontsize)
    axes[1].set_xlabel(
        r'$\mathregular{f \; [kHz]}$',
        fontsize=fontsize)
    axes[1].set_xlabel(
        'f [kHz]',
        fontsize=fontsize)
    labels = [
        r'$\mathregular{ECH \; @ \; \rho = 0.5}$',
        r'$\mathregular{ECH \; @ \; \rho = 0.8}$'
    ]
    axes[0].legend(
        labels,
        loc='lower left',
        fontsize=(fontsize - 2))

    # Limits and tick marks
    plt.xlim([10, 2000])
    axes[0].xaxis.set_major_formatter(
        mpl.ticker.FormatStrFormatter('%d'))
    axes[1].xaxis.set_major_formatter(
        mpl.ticker.FormatStrFormatter('%d'))
    axes[0].set_xticks([10, 100, 1000])
    axes[1].set_xticks([10, 100, 1000])

    ylim = [3e-13, 1e-7]
    axes[0].set_ylim(ylim)
    axes[1].set_ylim(ylim)
    axes[0].set_yticks([1e-12, 1e-10, 1e-8])

    # Indicate anti-aliasing filters
    axes[0].fill_betweenx(
        ylim,
        1e3,
        x2=2e3,
        color='lightgray')
    axes[0].text(
        1.3e3,
        2.5e-9,
        'anti-aliasing filters',
        rotation=90,
        fontsize=(fontsize - 2))

    # Add shot numbers & times
    x0 = 2.15e2

    sind = 0
    axes[1].annotate(
        '%i, [%.1f, %.1f] s' % (shots[sind], tlims[sind][0], tlims[sind][1]),
        (x0, 4.4e-8),
        color=cols[sind],
        fontsize=(fontsize - 6))
    sind = 1
    axes[1].annotate(
        '%i, [%.1f, %.1f] s' % (shots[sind], tlims[sind][0], tlims[sind][1]),
        (x0, 2.4e-8),
        color=cols[sind],
        fontsize=(fontsize - 6))

    plt.tight_layout()

    plt.show()
