import numpy as np
import matplotlib.pyplot as plt

import mitpci
import random_data as rd
import filters


shot = 1152
pci_channel = 8

# High-pass filter
hpf = filters.fir.Kaiser(
    -120, 5e2, 1e3, pass_zero=False, Fs=4e6)

# Spectral-estimation parameters
Tens = 0.05         # [Tens] = s
Nreal_per_ens = 10

# Unit conversions
Hz_per_kHz = 1e3

# Plotting parameters
flim = [3, 30]  # [flim] = Hz
fontsize = 15
annotation_color = 'white'
linewidth = 2
linestyle = '--'


def wavenumber(f_kHz, cs=343.):
    '''Return wavenumber of sound wave in cm^{-1}.

    Parameters:
    -----------
    f_kHz - float
        Frequency of sound wave.
        [f_kHz] = kHz

    cs - float
        Sound speed.
        [cs] = m/s

    Returns:
    --------
    k - float
        Wavenumber of sound wave
        [k] = cm^{-1}

    '''
    # Convert frequency to Hz and get wavenumber in m^{-1}
    k = 2 * np.pi * (f_kHz * 1e3) / cs

    # Convert to cm^{-1}
    k *= 1e-2

    return k


if __name__ == '__main__':
    # Load interferometer data and compute spectrum:
    # ----------------------------------------------
    L = mitpci.interferometer.Lissajous(shot)
    Ph_int = mitpci.interferometer.Phase(L, filt=hpf)
    asd_int = rd.spectra.AutoSpectralDensity(
        Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Load PCI data and compute spectrum:
    # -----------------------------------
    Ph_pci = mitpci.pci.Phase(shot, pci_channel, filt=hpf)
    asd_pci = rd.spectra.AutoSpectralDensity(
        Ph_pci.x, Fs=Ph_pci.Fs, t0=Ph_pci.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Unit conversions:
    # -----------------
    asd_int.f /= Hz_per_kHz
    asd_pci.f /= Hz_per_kHz

    asd_int.Gxx *= Hz_per_kHz
    asd_pci.Gxx *= Hz_per_kHz

    # Plotting:
    # ---------
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    xlabel = r'$\mathregular{t \; [s]}$'
    ylabel_f = r'$\mathregular{f \; [kHz]}$'
    ylabel_k = r'$\mathregular{k \; [cm^{-1}]}$'
    cblabel = r'$\mathregular{G_{\phi,\phi}(f) \; [rad^2 / kHz]}$'

    asd_int.plotSpectralDensity(
        flim=flim, ax=axes[0], title='interferometer',
        xlabel=xlabel, ylabel=ylabel_f, cblabel=cblabel,
        fontsize=fontsize)
    asd_pci.plotSpectralDensity(
        flim=flim, ax=axes[1], title='PCI',
        xlabel=xlabel, ylabel='', cblabel=cblabel,
        fontsize=fontsize)

    fmin, fmax = axes[1].get_ylim()
    kmin = wavenumber(fmin)
    kmax = wavenumber(fmax)

    ax1_k = axes[1].twinx()
    ax1_k.set_ylim(kmin, kmax)
    ax1_k.set_ylabel(ylabel_k, fontsize=fontsize)

    ax0_k = axes[0].twinx()
    ax0_k.set_ylim(kmin, kmax)
    ax0_k.set_yticklabels('')

    xlim = ax0_k.get_xlim()

    ax0_k.hlines(
        5., xlim[0], xlim[1],
        colors=annotation_color,
        linewidth=linewidth,
        linestyle=linestyle)
    ax0_k.text(
        0.1, 5.1, 'nominal interferometer cutoff',
        color=annotation_color,
        fontsize=(fontsize - 1))

    ax1_k.hlines(
        1.5, xlim[0], xlim[1],
        colors=annotation_color,
        linewidth=linewidth,
        linestyle=linestyle)
    ax1_k.text(
        1., 1.25, 'nominal PCI cutoff',
        color=annotation_color,
        fontsize=(fontsize - 1))
    ax1_k.text(
        4.3, 0.6, '%i' % shot,
        color=annotation_color,
        fontsize=(fontsize - 1))

    axes[0].set_xlim(xlim)  # Annotations mess up xlims, so fix it...

    plt.tight_layout()
    plt.show()
