import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct

import random_data as rd
import mitpci
import filters


# Spectral-estimation parameters
Tens = 0.25
Nreal_per_ens = 250

# FIR, zero-delay, high-pass filter
hpf = filters.fir.Kaiser(
    -120, 2.5e3, 5e3,
    pass_zero=False, Fs=4e6)

# Properties of delay line
v_RG58 = 2e8  # [v_RG58] = m/s; RG-58 coax has index of refraction ~3/2
L = 62.7      # [L] = m, from HWlogbook 1160411

# If True, attempt to compensate for old audio-amp rolloff
# for f > 300 kHz, which could distract from the point
# of the plot.
compensate_rolloff = True

# Dictionary with key: equivalent # of delay lines, value: shot
#
# What is meant by "equivalent # of delay lines"? For this test,
# we needed to maintain tau = 0 between the RF & LO lines, but
# we wanted to evaluate the effect of having a large amount of
# extra cabling on electrical pickup. To do this, we added equal
# amounts of extra cabling to both the RF & LO lines and then
# performed self-demodulation during Ops. Because the RF & LO
# each take at least one coax length to exit the pit, the
# configuration in which the RF & LO each have one cable
# corresponds to the equivalent of 0 delay lines. When the
# RF & LO each traverse 3 lines, we have an equivalent delay-line
# length of 2 * (3 - 1) = 4, and so on and so forth.
#
# Summary of shots w/ 5 cables each:
#
#     168945: I&Q sensible size, but enormous noise...
#     168946: I&Q sensible size, but enormous noise, sim. to 945
#     168947: No data.
#     168948: I&Q sensible size, but enormous noise, sim. to 945
#     168955: I&Q a bit small (?), but enormous quasi-coherent peak
#     168956: sim. to 955
#     168957: No data.
#     168958: I&Q too small to have good SNR...
#
shots = {
    0:  168960,  # 1 coax each
    4:  168962,  # 3 coax each
    12: 168923,  # 7 coax each
}

# Unit conversions
Hz_per_kHz = 1e3

# Plotting parameters
fontsize = 14
linewidth = 2
cols = get_distinct(len(shots.keys()))
flim = [10, 1e3]  # [flim] = kHz


if __name__ == '__main__':
    # Time delay corresponding to *single* delay line of length `L`
    # tau = (L / v_RG58) * 1e6  # [tau] = micro-s

    plt.figure()

    # Load and process LO self-demodulated shots
    # w/ varying number of delay lines; smallest to largest delay
    for i, lines in enumerate(np.sort(shots.keys())):
        shot = shots[lines]

        # Self demodulation produces (I,Q) confined to
        # a small neighborhood of full Lissajous ellipse,
        # so do *not* fit or compensate (I,Q).
        Liss = mitpci.interferometer.Lissajous(
            shot, fit=False, compensate=False)

        # Compute phase
        Ph = mitpci.interferometer.Phase(Liss, filt=hpf)

        # Estimate autospectral density
        asd = rd.spectra.AutoSpectralDensity(
            Ph.x, Fs=Ph.Fs, t0=Ph.t0,
            Tens=Tens, Nreal_per_ens=Nreal_per_ens)

        # Convert from Hz -> kHz
        asd.Gxx *= Hz_per_kHz
        asd.f /= Hz_per_kHz

        # Rolloff compensation; values from measurements
        if compensate_rolloff:
            alpha = 1.5  # Not sure why it's not integer...
            fc = 450.    # [fc] = kHz
            asd.Gxx *= (1 + ((asd.f[:, np.newaxis] / fc) ** alpha))

        # Determine indices for desired plot range:
        if i == 0:
            df = asd.f[1] - asd.f[0]
            find = np.where(np.logical_and(
                asd.f >= (flim[0] - df),
                asd.f <= (flim[-1] + df)))[0]

        # Construct label
        label = (
            r'$\mathregular{%i \; m \; (%i)}$'
            % (np.int(lines * L), shot))
        if i == 0:
            label = r'$\mathregular{L = }$' + label

        # Plot
        plt.loglog(
            asd.f[find],
            np.mean(asd.Gxx[find, :], axis=-1),
            linewidth=linewidth,
            c=cols[i],
            label=label)

    # Label plot
    plt.xlabel(
        r'$\mathregular{f \; [kHz]}$',
        fontsize=(fontsize + 2))
    plt.ylabel(
        r'$\mathregular{G_{\phi,\phi} \; [rad^2 / \, kHz]}$',
        fontsize=(fontsize + 2))
    plt.legend(loc='upper left')

    plt.xlim(flim)
    plt.ylim([3e-11, 1e-8])
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()
    plt.show()
