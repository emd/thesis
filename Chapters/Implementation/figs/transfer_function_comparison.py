import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct
from wavenumber_response import (
    PhaseContrastImaging, HeterodyneInterferometer)


# PCI parameters
kg = 1.5        # [kg] = cm^{-1}
kD = 75.        # [kD] = cm^{-1}
M_pci = 0.5     # approximately...
s_pci = 0.5e-1  # [s_pci] = cm

# Heterodyne interferometer parameters
M_het = 0.08
s_het = 1e-1    # [s_het] = cm

# Computational grid
kmax = 4 * kg
dk = 0.01
x_I_max = 1.
dx_I = 0.1

fontsize = 12
linewidth = 2


if __name__ == '__main__':
    # Create objects corresponding to each interferometric method
    pci = PhaseContrastImaging(kg=kg, kD=kD, M=M_pci, s=s_pci)
    int_het = HeterodyneInterferometer(M=M_het, s=s_het)

    # Compute wavenumber response of each interferometric method
    pci.applyTo(kmax=kmax, dk=dk, x_I_max=x_I_max, dx_I=dx_I)
    int_het.applyTo(kmax=kmax, dk=dk)

    # Determine index corresponding to PCI beam's center point
    # (in contrast, the homodyne and heterodyne interferometer
    # transfer functions are *independent* of the spatial coordinate)
    beam_center = len(pci.x_I) // 2

    # Also, PCI transfer function only defined for |k| >= kg
    hi_kind_pos = np.where(pci.k >= kg)[0]
    hi_kind_neg = np.where(pci.k <= -kg)[0]
    lo_kind = np.where(np.abs(pci.k) < kg)[0]

    # Color-blind proof colors
    pci_col, het_col = get_distinct(2)

    fig = plt.figure()

    # PCI transfer function
    plt.semilogy(
        pci.k[hi_kind_neg],
        np.abs(pci.A_pci[hi_kind_neg, :][:, beam_center]),
        c=pci_col,
        linewidth=linewidth,
        label='PCI')
    plt.semilogy(
        pci.k[hi_kind_pos],
        np.abs(pci.A_pci[hi_kind_pos, :][:, beam_center]),
        c=pci_col,
        linewidth=linewidth)

    # Heterodyne interferometer transfer function
    plt.semilogy(
        int_het.k,
        int_het.Thet,
        c=het_col,
        linewidth=linewidth,
        label='heterodyne int.')

    # PCI low-k amplitude response
    plt.semilogy(
        pci.k[lo_kind],
        np.abs(pci.A_pci[lo_kind, :][:, beam_center]),
        c=pci_col,
        linewidth=linewidth,
        linestyle='-.')

    # PCI low-k cutoff
    plt.axvline(x=pci.kg, c='k', linewidth=linewidth, linestyle='--')
    plt.axvline(x=-pci.kg, c='k', linewidth=linewidth,  linestyle='--')

    plt.xlim([-kmax, kmax])
    plt.ylim([1e-2, 3e1])

    plt.xlabel('$k \; [\mathrm{cm}^{-1}]$', fontsize=fontsize)
    plt.ylabel('$T(k)$ [unitless]', fontsize=fontsize)

    plt.legend(loc='upper right')

    plt.show()
