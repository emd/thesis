import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct

from measured_noise import (
    quantization_noise_spectral_density,
    self_demodulation_spectral_density,
    full_system_calibration_spectral_density,
    Lmode_spectral_density,
    ELM_free_Hmode_spectral_density)

from expected_noise import (
    expected_quantization_noise_spectral_density,
    detector_noise_spectral_density,
    optical_shot_noise_spectral_density,
    LO_instrumental_phase_noise_spectral_density)


# Plotting parameters
figsize = (8, 7)
linewidth = 2
measured_linestyle = '-'
predicted_linestyle = '--'
cols = get_distinct(5)[::-1]
fontsize = 14
flim = [10, 1000]   # [flim] = kHz

# Spectral-estimation parameters (sets frequency resolution)
Tens = 0.05         # [Tens] = s
Nreal_per_ens = 50

# Compensate rolloff of audio amps in certain shots?
compensate_rolloff = True


if __name__ == '__main__':
    # Spectral computations:
    # ======================

    # Measured noise spectral densities:
    # ----------------------------------
    res = quantization_noise_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    f = res[0]
    Gxx_quant = res[1]
    shot_quant = res[2]

    res = self_demodulation_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens,
        compensate_rolloff=compensate_rolloff)
    f = res[0]
    Gxx_electronics = res[1]
    shot_electronics = res[2]

    res = full_system_calibration_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    f = res[0]
    Gxx_full_system = res[1]
    shot_full_system = res[2]

    # Expected noise spectral densities:
    # ----------------------------------
    Gxx_quant_expected = expected_quantization_noise_spectral_density(
        Nf=len(f))

    Gxx_detector = detector_noise_spectral_density(Nf=len(f))

    Gxx_shot_noise = optical_shot_noise_spectral_density(Nf=len(f))

    Gxx_LO = LO_instrumental_phase_noise_spectral_density(f)

    # Measured plasma-fluctuation spectral densities:
    # -----------------------------------------------
    res = Lmode_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    f = res[0]
    Gxx_Lmode = res[1]
    shot_Lmode = res[2]

    res = ELM_free_Hmode_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    f = res[0]
    Gxx_ELM_free_Hmode = res[1]
    shot_ELM_free_Hmode = res[2]

    # Plotting:
    # =========
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    # Determine indices for desired plot range:
    # -----------------------------------------
    df = f[1] - f[0]
    find = np.where(np.logical_and(
        f >= (flim[0] - df),
        f <= (flim[-1] + df)))[0]

    # Predictions:
    # ------------
    ax.text(50., 6e-15, 'optical shot noise (P)', fontsize=fontsize)
    ax.loglog(
        f,
        Gxx_shot_noise,
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c='k')

    ax.text(25., 1.75e-13,
        'LO w/ $\mathregular{\\tau = 2.5 \, \mu s}$ (P)',
        fontsize=fontsize,
        rotation=22.5)
    ax.loglog(
        f[find],
        Gxx_LO[find],
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c='k')

    ax.text(11., 1.5e-12, 'bit noise (P)', fontsize=fontsize)
    ax.loglog(
        f[find],
        Gxx_quant_expected[find],
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c='k')

    ax.text(11., 7e-12, 'detector noise (P)', fontsize=fontsize)
    ax.loglog(
        f,
        Gxx_detector,
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c='k')

    # Measured:
    # ---------
    cind = 0
    ax.loglog(
        f[find],
        Gxx_quant[find],
        label='bit noise (%i)' % shot_quant,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    cind += 1
    ax.loglog(
        f[find],
        Gxx_electronics[find],
        label='all electronics (%i)' % shot_electronics,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    cind += 1
    ax.loglog(
        f[find],
        Gxx_full_system[find],
        label='full system (%i)' % shot_full_system,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    cind += 1
    ax.loglog(
        f[find],
        Gxx_Lmode[find],
        label='L-mode (%i)' % shot_Lmode,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    cind += 1
    ax.loglog(
        f[find],
        Gxx_ELM_free_Hmode[find],
        label='H-mode (%i)' % shot_ELM_free_Hmode,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    # Labeling & fine-tuning:
    # -----------------------
    plt.xlabel(
        '$\mathregular{f \; [kHz]}$',
        fontsize=(fontsize + 2))
    plt.ylabel(
        '$\mathregular{G_{\phi,\phi}(f) \; [rad^2 / kHz]}$',
        fontsize=(fontsize + 2))

    plt.xlim(flim)
    plt.ylim([1e-15, 1e-6])
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.legend(ncol=2, loc='best')

    plt.show()
