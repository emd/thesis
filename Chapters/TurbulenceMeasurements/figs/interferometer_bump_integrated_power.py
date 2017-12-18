import numpy as np
import matplotlib.pyplot as plt
import random_data as rd
import mitpci
from gadata import gadata
from distinct_colours import get_distinct


# Windows for analysis: includes shot, time window, & ECH location
from windows import windows

# ELM-filtering parameters
sigma_mult = 3
debounce_dt = 0.5e-3            # Less than ELM spacing
window_fraction = [0.2, 0.8]

pci_channel = 8

# Spectral parameters
Tens = 0.5e-3       # Less than ELM spacing
Nreal_per_ens = 1   # Do averaging after ELM-filtering

# Remove interferometer coherent pickup
fcoherent = [
    [321993, 345681],
    [490328, 514368],
    [654570, 682260],
    [825666, 850316],
    [990271, 1017980]
]

bump_bandwdith = [300e3, 600e3]

# Plotting parameters
figsize = (6, 4)
fontsize = 15
marker = 'o'
cols = get_distinct(2)
fit_color = 'k'
fit_linestyle = '--'
linewidth = 2


if __name__ == '__main__':
    Nwin = len(windows)
    varphi = np.zeros(Nwin)
    rho_ECH = np.zeros(Nwin)

    for window_index, w in enumerate(windows):
        print '\nWindow %i of %i' % (window_index + 1, Nwin)

        rho = w.rho_ECH
        shot = w.shot
        tlim = w.tlim

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

        # Find ELMs & average only over inter-ELM window
        SH = rd.signals.SpikeHandler(
            Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
            sigma_mult=sigma_mult, debounce_dt=debounce_dt)
        tind = SH.getSpikeFreeTimeIndices(
            asd_int.t, window_fraction=window_fraction)

        Gxx_int = np.mean(asd_int.Gxx[:, tind], axis=-1)
        Gxx_pci = np.mean(asd_pci.Gxx[:, tind], axis=-1)

        # Plot ELM-filtered spectra over specified frequency range
        fig, axes = plt.subplots(2, 1, sharex=True)

        find_int = np.where(np.logical_and(
            asd_int.f >= 10e3,
            asd_int.f <= 1e6))[0]
        find_pci = np.where(np.logical_and(
            asd_pci.f >= 10e3,
            asd_pci.f <= 2e6))[0]

        axes[0].loglog(
            asd_int.f[find_int],
            Gxx_int[find_int])
        axes[1].loglog(
            asd_pci.f[find_pci],
            Gxx_pci[find_pci])
        axes[1].set_xlabel('f [Hz]')
        axes[0].set_ylabel('S(f) [$\mathregular{rad^2}$/ Hz]')
        axes[1].set_ylabel('S(f) [$\mathregular{rad^2}$/ Hz]')
        axes[0].set_title('interferometer')
        axes[1].set_title('PCI')

        # Note bump bandwidth
        axes[0].vlines(
            bump_bandwdith,
            1e-13,
            1e-9,
            linestyle='--',
            color='k')

        plt.tight_layout()
        fname = ('./Sf_rho%.1f_%i_%.1f_to_%.1f.pdf'
                 % (rho, shot, tlim[0], tlim[1]))
        plt.savefig(fname)

        plt.close()

        # Integrate over bump bandwidth
        find_bump = np.where(np.logical_and(
            asd_int.f >= bump_bandwdith[0],
            asd_int.f <= bump_bandwdith[1]))[0]
        varphi[window_index] = np.sum(Gxx_int[find_bump]) * asd_int.df
        rho_ECH[window_index] = rho

    # Convert varphi to mrad^2
    varphi *= 1e6

    # Fit a line to data, just to show there is indeed a trend
    A = np.array([rho_ECH, np.ones(len(rho_ECH))]).T

    fit = np.linalg.lstsq(A, varphi)[0]
    m = fit[0]
    b = fit[1]

    # Plot integrated powers
    plt.figure(figsize=figsize)

    rho_array = np.arange(0, 1.1, 0.1)
    plt.plot(
        rho_array,
        (m * rho_array) + b,
        c=fit_color,
        linewidth=linewidth,
        linestyle=fit_linestyle)

    ind05 = np.where(rho_ECH == 0.5)
    ind08 = np.where(rho_ECH == 0.8)

    plt.plot(
        rho_ECH[ind05],
        varphi[ind05],
        marker,
        c=cols[0])
    plt.plot(
        rho_ECH[ind08],
        varphi[ind08],
        marker,
        c=cols[1])

    plt.xlabel(
        r'$\mathregular{\rho_{ECH}}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{var(\widetilde{\phi}) \; [mrad^2]}$',
        fontsize=fontsize)

    plt.xlim([0, 1])
    plt.ylim([0, 0.1])

    plt.tight_layout()

    plt.show()
