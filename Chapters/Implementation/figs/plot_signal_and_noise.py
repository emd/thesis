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
figsize = (11, 7)
linewidth = 2
measured_linestyle = '-'
predicted_linestyle = '-'
cols = get_distinct(8)[::-1]
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

    # Optical shot noise:
    # -------------------
    cind = 0

    ax.loglog(
        f,
        Gxx_shot_noise,
        label='optical shot noise (P)',
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c=cols[cind])

    # LO:
    # ---
    cind += 1

    ax.loglog(
        f[find],
        Gxx_LO[find],
        label='LO w/ $\mathregular{\\tau = 2.5 \, \mu s}$ (P)',
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c=cols[cind])

    # Bit noise:
    # ----------
    cind += 1

    ax.loglog(
        f[find],
        Gxx_quant_expected[find],
        label='bit noise (P)',
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c=cols[cind])

    ax.loglog(
        f[find],
        Gxx_quant[find],
        label='bit noise (%i)' % shot_quant,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    # Electronics:
    # ------------
    cind += 1

    ax.loglog(
        f[find],
        Gxx_electronics[find],
        label='all electronics (%i)' % shot_electronics,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    # Detector:
    # ---------
    cind += 1

    ax.loglog(
        f,
        Gxx_detector,
        label='detector (P)',
        linewidth=linewidth,
        linestyle=predicted_linestyle,
        c=cols[cind])

    # Full system:
    # ------------
    cind += 1

    ax.loglog(
        f[find],
        Gxx_full_system[find],
        label='full system (%i)' % shot_full_system,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    # L-mode:
    # -------
    cind += 1

    ax.loglog(
        f[find],
        Gxx_Lmode[find],
        label='L-mode (%i)' % shot_Lmode,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    # H-mode:
    # -------
    cind += 1

    ax.loglog(
        f[find],
        Gxx_ELM_free_Hmode[find],
        label='H-mode (%i)' % shot_ELM_free_Hmode,
        linewidth=linewidth,
        linestyle=measured_linestyle,
        c=cols[cind])

    # Labeling:
    # ---------
    plt.xlabel(
        '$\mathregular{f \; [kHz]}$',
        fontsize=fontsize)
    plt.ylabel(
        '$\mathregular{G_{\phi,\phi}(f) \; [rad^2 / kHz]}$',
        fontsize=fontsize)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlim(flim)
    plt.ylim([1e-15, 1e-7])

    plt.show()
