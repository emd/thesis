import numpy as np
import matplotlib.pyplot as plt
from wavenumber_response import PhaseContrastImaging
import random_data as rd


# Optical parameters of PCI system
kg = 2.  # [kg] = 1 / w0;  kg = 2 is the diffraction limit
M = 0.5  # [M] = unitless; typical value...

# Computational grid
kmax = 4 * kg       # [kmax] = [kg]
dk = 0.05           # [dk] = [kg]

xmax = 1.5          # [xmax] = w0
dx = 0.01           # [dx] = w0
xmax_I = xmax * M   # [xmax_I] = M * w0
dx_I = dx * M       # [dx_I] = M * w0

# Spectral-estimation parameters
p = 2
Nk = 100000  # Larger values lead to a smoother plot

# Plotting parameters
fontsize = 12
linewidth = 2
xlim = [-kmax, kmax]
ylim = [-kmax, kmax]


if __name__ == '__main__':
    pci = PhaseContrastImaging(kg=kg, M=M, s=1)
    pci.applyTo(kmax=kmax, dk=dk, x_I_max=xmax_I, dx_I=dx_I)

    Fs_spatial = 1. / (pci.x[1] - pci.x[0])
    kmeas = np.zeros(len(pci.k))

    for kind, k in enumerate(pci.k):
        # Phase-response contribution to signal's spatial structure.
        # This approach allows us to figure out the shift `dk` to the
        # true wavenumber `k`, i.e. `kmeas = k + dk`.
        #
        # (Note that something weird happens with the spectral estimation
        # when we use the full phase of (1) the physical phase k * x and
        # (2) the PCI phase response -- instead of linear response well
        # outside of |k| > k_g, we get a "wavy" response... Even removing
        # contribution (2) and only using the physical phase (1) produces
        # something similar, which tells me that there is something weird
        # happening with the Burg spectral estimation... The approach used
        # here avoids these problems and displays the nonlinearity for
        # |k| < k_g).
        y = np.cos(pci.theta_pci[kind, :])

        # Compute autospectral density
        asd = rd.spectra.parametric.BurgAutoSpectralDensity(
            p, y, Fs=Fs_spatial, Nf=Nk, normalize=False)

        # Find peak in autospectral density
        if k == 0:
            # By symmetry arguments
            dk = 0
        else:
            if k < 0:
                sl = slice(0, len(asd.f) // 2 + 1)
            else:
                sl = slice(len(asd.f) // 2, None)

            maxind = np.where(asd.Sxx[sl] == np.max(asd.Sxx[sl]))[0]

            if len(maxind) == 0:
                # This occurs if `pci.theta_pci[kind, :]` is very nearly zero,
                # such that `asd.Sxx` is `nan` and we obtain no shift in
                # the measured wavenumber
                dk = 0
            else:
                dk = 2 * np.pi * asd.f[sl.start + maxind]

        kmeas[kind] = k + dk

    # Plotting
    plt.figure()

    plt.plot(pci.k, kmeas, linewidth=linewidth)
    plt.axvline(kg, c='k', linestyle='--', linewidth=linewidth)
    plt.axvline(-kg, c='k', linestyle='--', linewidth=linewidth)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal')

    plt.xlabel('$k \; [1 /\, w_0]$', fontsize=fontsize)
    plt.ylabel('$k_{\mathrm{meas}} \; [1 /\, w_0]$', fontsize=fontsize)

    plt.show()
