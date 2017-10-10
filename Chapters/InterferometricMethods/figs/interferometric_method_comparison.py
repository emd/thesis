import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct
from wavenumber_response import (
    PhaseContrastImaging, HeterodyneInterferometer, HomodyneInterferometer)


# Optical parameters of PCI system
kg = 2.  # kg = 2 / w0 is the diffraction limit
M = 0.5  # typical...

# Detector size
s = 0  # Don't include finite sampling-volume effects here

# Computational grid
kmax = 4 * kg
dk = 0.01
x_I_max = 1.
dx_I = 0.1

fontsize = 12
linewidth = 2


if __name__ == '__main__':
    # Create objects corresponding to each interferometric method
    pci = PhaseContrastImaging(kg=kg, M=M, s=s)
    int_het = HeterodyneInterferometer(M=M, s=s)
    int_hom_opt = HomodyneInterferometer(M=M, s=s, dphi=(np.pi / 2))
    # int_hom_half_opt = HomodyneInterferometer(M=M, s=s, dphi=(np.pi / 4))
    # int_hom_quart_opt = HomodyneInterferometer(M=M, s=s, dphi=(np.pi / 8))

    # Compute wavenumber response of each interferometric method
    pci.applyTo(kmax=kmax, dk=dk, x_I_max=x_I_max, dx_I=dx_I)
    int_het.applyTo(kmax=kmax, dk=dk)
    int_hom_opt.applyTo(kmax=kmax, dk=dk)
    # int_hom_half_opt.applyTo(kmax=int_hom_half_opt.kfsv, dk=dk)
    # int_hom_quart_opt.applyTo(kmax=int_hom_quart_opt.kfsv, dk=dk)

    # Determine index corresponding to PCI beam's center point
    # (in contrast, the homodyne and heterodyne interferometer
    # transfer functions are *independent* of the spatial coordinate)
    beam_center = len(pci.x_I) // 2

    # Color-blind proof colors
    pci_col, het_col, hom_col = get_distinct(3)

    fig = plt.figure()

    plt.semilogy(pci.k, np.abs(pci.A_pci[:, beam_center]),
                 c=pci_col, linewidth=linewidth)
    plt.semilogy(int_het.k, int_het.Thet,
                 c=het_col, linewidth=linewidth)
    plt.semilogy(int_hom_opt.k, int_hom_opt.Thom,
                 c=hom_col, linewidth=linewidth, linestyle='-')
    # plt.semilogy(int_hom_half_opt.k, int_hom_half_opt.Thom,
    #              c=hom_col, linewidth=linewidth, linestyle='--')
    # plt.semilogy(int_hom_quart_opt.k, int_hom_quart_opt.Thom,
    #              c=hom_col, linewidth=linewidth, linestyle=':')

    plt.axvline(x=pci.kg, c='k', linewidth=linewidth, linestyle='--')
    plt.axvline(x=-pci.kg, c='k', linewidth=linewidth,  linestyle='--')

    plt.xlim([-kmax, kmax])
    plt.ylim([1e-2, 3e1])

    plt.xlabel('$k \; [1 /\, w_0]$', fontsize=fontsize)
    plt.ylabel('$T(k)$', fontsize=fontsize)
    # plt.title('interferometric method transfer functions')

    plt.legend([
        'PCI (x = 0)',
        'heterodyne int.',
        'homodyne int.'],  # $\phi_R - \\bar{\phi} = \pi / 2$'],
        loc='upper right')

    plt.show()
