import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct

import mitpci
import random_data as rd
import filters
from wavenumber import wavenumber
from tymphany_model import phase_shift_ideal_system


shot = 1152

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
cols = get_distinct(2)
annotation_color = 'white'
annotation_linestyle = '--'
noise_linestyle = '--'


def interferometer_expected_response(k_prof, noise_prof, kfsv=5., dk=1e-2):
    # Convert k-range [cm^{-1}] into corresponding frequencies [kHz]
    cs = 343.  # [cs] = m / s
    fmin = (0.1 / (2 * np.pi)) * cs * k_prof[0]
    fmax = (0.1 / (2 * np.pi)) * cs * k_prof[-1]
    df = (0.1 / (2 * np.pi)) * cs * dk
    freqs = np.arange(fmin, fmax + df, df)

    # Compute expected phase shift for *ideal system*
    # and parse results
    res = phase_shift_ideal_system(freqs=freqs)
    k = res[0]
    varphi_min = res[1]
    varphi_max = res[2]

    # Account for finite sampling-volume effects;
    # note that sinc weighting is for *amplitude*, and
    # *power* is weighted with sinc^2
    w = (np.sinc(k / kfsv)) ** 2
    varphi_min *= w
    varphi_max *= w

    # Accurately predicting the interferometer response
    # requires accounting for the system noise floor,
    # particularly in regions with low SNR.
    #
    # Interpolate the measured noise profile onto
    # the wavenumber grid `k`.
    noise_prof_interp = np.interp(k, k_prof, noise_prof)

    varphi_min += noise_prof_interp
    varphi_max += noise_prof_interp

    return k, varphi_min, varphi_max


if __name__ == '__main__':
    # Load interferometer data and compute spectrum:
    # ----------------------------------------------
    L = mitpci.interferometer.Lissajous(shot)
    Ph_int = mitpci.interferometer.Phase(L, filt=hpf)
    asd_int = rd.spectra.AutoSpectralDensity(
        Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Unit conversions:
    # -----------------
    asd_int.f /= Hz_per_kHz

    asd_int.Gxx *= Hz_per_kHz

    # Plot spectrogram:
    # -----------------
    xlabel = r'$\mathregular{t \; [s]}$'
    ylabel_f = r'$\mathregular{f \; [kHz]}$'
    ylabel_k = r'$\mathregular{k \; [cm^{-1}]}$'
    cblabel = r'$\mathregular{G_{\phi,\phi}(f) \; [rad^2 / kHz]}$'

    asd_int.plotSpectralDensity(
        flim=flim, title='interferometer',
        xlabel=xlabel, ylabel=ylabel_f, cblabel=cblabel,
        fontsize=fontsize)

    ax = plt.gca()
    fmin, fmax = ax.get_ylim()
    kmin = wavenumber(fmin)
    kmax = wavenumber(fmax)

    ax_k = ax.twinx()
    ax_k.set_ylim(kmin, kmax)
    ax_k.set_yticklabels('')

    plt.tight_layout()
    plt.show()

    # Compute line profile from sound wave:
    # -------------------------------------
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    src_soundwave = (2.5, 0.876)
    dst_soundwave = (27.9, 2.368)
    dt_noise = 0.5

    df = 1.5  # [df] = kHz
    lpc_kwargs = {'N': 100, 'lwr': df, 'L': 11}
    coord_lines = rd.utilities.line_profile_coordinates(
        src_soundwave, dst_soundwave, **lpc_kwargs)

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
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Response profile
    Gxx_int_prof, f_prof, t_prof = rd.utilities.line_profile(
        asd_int.Gxx, asd_int.f, asd_int.t,
        src_soundwave, dst_soundwave,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 3})

    # Noise profile
    Gxx_int_noise_prof, f_prof, t_prof = rd.utilities.line_profile(
        asd_int.Gxx, asd_int.f, asd_int.t + dt_noise,
        src_soundwave, dst_soundwave,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 3})

    # The profiles are *averages* over frequency, so
    # multiply by bandwidth to get integrated power
    Gxx_int_prof *= df
    Gxx_int_noise_prof *= df

    k_prof = wavenumber(f_prof)
    kind = np.where(np.logical_and(
        k_prof >= klim[0],
        k_prof <= klim[1]))[0]

    # Compute expected response profile:
    # ----------------------------------
    res = interferometer_expected_response(
        k_prof[kind], Gxx_int_noise_prof[kind],
        kfsv=5., dk=1e-2)
    k_expected = res[0]
    varphi_min_expected = res[1]
    varphi_max_expected = res[2]

    # Plot response profile:
    # ----------------------
    plt.figure()

    plt.fill_between(
        k_expected,
        varphi_min_expected,
        varphi_max_expected,
        color=cols[0],
        label='expected signal')
    plt.semilogy(
        k_prof[kind],
        Gxx_int_prof[kind],
        c='k',  # cols[1],
        linewidth=linewidth,
        label='measured signal')
    plt.semilogy(
        k_prof[kind],
        Gxx_int_noise_prof[kind],
        c='k',  # cols[1],
        linewidth=linewidth,
        linestyle=noise_linestyle,
        label='measured noise')

    plt.xlabel(
        r'$\mathregular{k \; [cm^{-1}]}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{var\,(\widetilde{\phi}) \;\, [rad^2]}$',
        fontsize=fontsize)
    plt.legend(loc='best')

    plt.xlim(klim)
    plt.tick_params(
        axis='both',
        which='major',
        labelsize=(fontsize - 2))

    plt.show()
