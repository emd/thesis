import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct

import mitpci
import random_data as rd
import filters
from wavenumber import wavenumber


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
flim = [3, 30]        # [flim] = Hz
klim = [0.5, 4.5]     # [klim] = cm^{-1}
fontsize = 15
linewidth = 2
signal_linewidth = linewidth + 2
cols = get_distinct(2)
annotation_color = 'white'
annotation_linestyle = '--'
noise_linestyle = '--'
cutoff_dashes = [12, 4, 2, 4]


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

    # Plot spectrograms:
    # ------------------
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    xlabel = r'$\mathregular{t \; [s]}$'
    ylabel_f = r'$\mathregular{f \; [kHz]}$'
    ylabel_k = r'$\mathregular{k \; [cm^{-1}]}$'
    cblabel = r'$\mathregular{G_{\phi,\phi}(f) \; [rad^2 / kHz]}$'

    asd_int.plotSpectralDensity(
        flim=flim, ax=axes[0], title='interferometer',
        # vlim=[1e-9, 1e-7],
        xlabel=xlabel, ylabel=ylabel_f, cblabel=cblabel,
        fontsize=fontsize)
    asd_pci.plotSpectralDensity(
        flim=flim, ax=axes[1], title='PCI',
        # vlim=[1e-11, 1e-7],
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

    plt.tight_layout()
    plt.show()

    # Compute line profile from sound wave
    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()

    src_soundwave = (2.5, 0.876)
    dst_soundwave = (27.9, 2.368)
    dt_noise = 0.5

    # lpc_kwargs = {'N': 50, 'lwc': 2 * asd_int.dt, 'L': 11}
    df = 1.5  # [df] = kHz
    lpc_kwargs = {'N': 100, 'lwr': df, 'L': 11}
    coord_lines = rd.utilities.line_profile_coordinates(
        src_soundwave, dst_soundwave, **lpc_kwargs)

    for ax in axes:
        # Plot boundary for signal
        ax.plot(
            coord_lines[1, :, 0],
            coord_lines[0, :, 0],
            c=annotation_color,
            linestyle=annotation_linestyle,
            linewidth=linewidth)
        ax.plot(
            coord_lines[1, :, -1],
            coord_lines[0, :, -1],
            c=annotation_color,
            linestyle=annotation_linestyle,
            linewidth=linewidth)

        # Plot boundary for noise
        ax.plot(
            coord_lines[1, :, 0] + dt_noise,
            coord_lines[0, :, 0],
            c=annotation_color,
            linestyle=annotation_linestyle,
            linewidth=linewidth)
        ax.plot(
            coord_lines[1, :, -1] + dt_noise,
            coord_lines[0, :, -1],
            c=annotation_color,
            linestyle=annotation_linestyle,
            linewidth=linewidth)

    # Annotating plots with line-profile boundaries messes up plot range...
    # so fix it manually
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)

    # Response profiles
    Gxx_int_prof, f_prof, t_prof = rd.utilities.line_profile(
        asd_int.Gxx, asd_int.f, asd_int.t,
        src_soundwave, dst_soundwave,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 3})
    Gxx_pci_prof, f_prof, t_prof = rd.utilities.line_profile(
        asd_pci.Gxx, asd_pci.f, asd_pci.t,
        src_soundwave, dst_soundwave,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 3})

    # Noise profiles
    Gxx_int_noise_prof, f_prof, t_prof = rd.utilities.line_profile(
        asd_int.Gxx, asd_int.f, asd_int.t + dt_noise,
        src_soundwave, dst_soundwave,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 3})
    Gxx_pci_noise_prof, f_prof, t_prof = rd.utilities.line_profile(
        asd_pci.Gxx, asd_pci.f, asd_pci.t + dt_noise,
        src_soundwave, dst_soundwave,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 3})

    # The profiles are *averages* over frequency, so
    # multiply by bandwidth to get integrated power
    Gxx_int_prof *= df
    Gxx_pci_prof *= df
    Gxx_int_noise_prof *= df
    Gxx_pci_noise_prof *= df

    k_prof = wavenumber(f_prof)

    # Interferometer finite sampling-volume weighting
    w_int = (np.sinc(k_prof / 5.)) ** 2

    # Define various index subsets for plotting
    kind_klim = np.where(np.logical_and(
        k_prof >= klim[0],
        k_prof <= klim[1]))[0]

    k_cutoff_int = 3.5
    kind_int = np.where(np.logical_and(
        k_prof >= klim[0],
        k_prof <= k_cutoff_int))[0]
    kind_int_cutoff = np.where(np.logical_and(
        k_prof >= k_cutoff_int,
        k_prof <= klim[1]))[0]

    k_cutoff_pci = 2.
    kind_pci = np.where(np.logical_and(
        k_prof >= k_cutoff_pci,
        k_prof <= klim[1]))[0]
    kind_pci_cutoff = np.where(np.logical_and(
        k_prof >= klim[0],
        k_prof <= k_cutoff_pci))[0]

    # Plot
    plt.figure()

    plt.semilogy(
        k_prof[kind_int],
        Gxx_int_prof[kind_int] / w_int[kind_int],
        c=cols[0],
        linewidth=signal_linewidth,
        label='interferometer signal')
    plt.semilogy(
        k_prof[kind_int_cutoff],
        Gxx_int_prof[kind_int_cutoff] / w_int[kind_int_cutoff],
        c=cols[0],
        linewidth=linewidth,
        dashes=cutoff_dashes)
    plt.semilogy(
        k_prof[kind_klim],
        Gxx_int_noise_prof[kind_klim],
        c=cols[0],
        linewidth=linewidth,
        linestyle=noise_linestyle,
        label='interferometer noise')

    plt.semilogy(
        k_prof[kind_pci],
        Gxx_pci_prof[kind_pci],
        c=cols[1],
        linewidth=signal_linewidth,
        label='PCI signal')
    plt.semilogy(
        k_prof[kind_pci_cutoff],
        Gxx_pci_prof[kind_pci_cutoff],
        c=cols[1],
        linewidth=linewidth,
        dashes=cutoff_dashes)
    plt.semilogy(
        k_prof[kind_klim],
        Gxx_pci_noise_prof[kind_klim],
        c=cols[1],
        linewidth=linewidth,
        linestyle=noise_linestyle,
        label='PCI noise')

    plt.xlabel(
        r'$\mathregular{k \; [cm^{-1}]}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{var\,(\widetilde{\phi}) \;\, [rad^2]}$',
        fontsize=fontsize)
    plt.legend(ncol=2, loc='best')

    plt.text(4.15, 5e-7, '%i' % shot, fontsize=(fontsize - 2))

    plt.xlim(klim)
    plt.tick_params(
        axis='both',
        which='major',
        labelsize=(fontsize - 2))

    plt.show()
