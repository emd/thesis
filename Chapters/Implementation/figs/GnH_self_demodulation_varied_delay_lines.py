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

# Dictionary with key: test shot, value: # of delay lines
delay_lines = {
    1168: 8,
    1169: 6,
    1170: 4,
    1171: 2,
    1172: 0,
    # 1173: 0.5,
    # 1174: 1
}

# Properties of delay line
v_RG58 = 2e8  # [v_RG58] = m/s; RG-58 coax has index of refraction ~3/2
L = 62.7      # [L] = m, from HWlogbook 1160411

# Unit conversions
Hz_per_kHz = 1e3

# Plotting parameters
fontsize = 14
linewidth = 2
cols = get_distinct(len(delay_lines.keys()))
flim = [10, 1e3]  # [flim] = kHz


if __name__ == '__main__':
    # Time delay corresponding to *single* delay line of length `L`
    tau = (L / v_RG58) * 1e6  # [tau] = micro-s

    plt.figure()

    # Load and process LO self-demodulated shots
    # w/ varying number of delay lines; smallest to largest delay
    for i, shot in enumerate(np.sort(delay_lines.keys())[::-1]):
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

        # Determine indices for desired plot range:
        if i == 0:
            df = asd.f[1] - asd.f[0]
            find = np.where(np.logical_and(
                asd.f >= (flim[0] - df),
                asd.f <= (flim[-1] + df)))[0]

        # Construct label
        label = (
            r'$\mathregular{%.1f \; \mu s \; (%i)}$'
            % (delay_lines[shot] * tau, shot))
        if i == 0:
            label = r'$\mathregular{\tau = }$' + label

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
    plt.legend(loc='upper right')

    plt.xlim(flim)
    plt.ylim([3e-11, 3e-6])
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()
    plt.show()
