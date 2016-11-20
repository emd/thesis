import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct
from wavenumber_response import (
    PhaseContrastImaging, HeterodyneInterferometer, HomodyneInterferometer)


# Optical parameters of PCI system
kg = 2.  # kg = 2 / w0 is the diffraction limit
M = 0.5  # typical...

# Detector size
kfsv_desired = 30.                # [kfsv_desired] = 1 / [w0]
s = 2 * np.pi * M / kfsv_desired  # gives s ~ 0.1 * (M * w0), which is typical
print 'detector element size in image plane, s = %f w0\n' % s

# Computational grid
kmax = 5 * kg
dk = 0.01
x_I_max = 1.
dx_I = 0.1

fontsize = 16
linewidth = 2


if __name__ == '__main__':
    # Create objects corresponding to each interferometric method
    pci = PhaseContrastImaging(kg=kg, M=M, s=s)
    pci_finite_kD = PhaseContrastImaging(
        kg=kg, kD=((2. / 3) * pci.kfsv), M=M, s=s)

    # Compute wavenumber response of each interferometric method
    pci.applyTo(kmax=pci.kfsv, dk=dk, x_I_max=x_I_max, dx_I=dx_I)
    pci_finite_kD.applyTo(kmax=pci.kfsv, dk=dk, x_I_max=x_I_max, dx_I=dx_I)

    # Determine index corresponding to PCI beam's center point
    # (in contrast, the homodyne and heterodyne interferometer
    # transfer functions are *independent* of the spatial coordinate)
    beam_center = len(pci.x_I) // 2

    # Color-blind proof colors
    cols = get_distinct(3)

    fig = plt.figure()

    plt.semilogy(pci.k, np.abs(pci.Tpp[:, beam_center]),
                 linewidth=linewidth, c=cols[0])
    plt.semilogy(pci.k, np.abs(pci.Tpci[:, beam_center]),
                 linewidth=linewidth, c=cols[1])
    plt.semilogy(pci.k, np.abs(pci_finite_kD.Tpci[:, beam_center]),
                 linewidth=linewidth, c=cols[2])

    plt.axvline(x=pci.kg, c='k', linewidth=linewidth, linestyle='--')
    plt.axvline(x=-pci.kg, c='k', linewidth=linewidth, linestyle='--')
    plt.axvline(
        x=pci_finite_kD.kD, c='k', linewidth=linewidth, linestyle='-.')
    plt.axvline(
        x=-pci_finite_kD.kD, c='k', linewidth=linewidth, linestyle='-.')

    plt.ylim([1e-2, 1e2])

    plt.xlabel('$k \, [1 / w_0]$', fontsize=fontsize)
    plt.ylabel('$T(k)$', fontsize=fontsize)
    plt.title('effects of various PCI cutoffs')

    plt.legend([
        '$T_{\mathrm{pp}}(k)$',
        '$T_{\mathrm{pci}}(k), \, k_{\mathrm{fsv}} = 30, k_D = \infty$',
        '$T_{\mathrm{pci}}(k), \, k_{\mathrm{fsv}} = 30, k_D = 20$'],
        loc='upper right')

    plt.show()
