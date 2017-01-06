import numpy as np
import matplotlib.pyplot as plt
from wavenumber_response import PhaseContrastImaging


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
dx_I = 0.01


if __name__ == '__main__':
    pci = PhaseContrastImaging(kg=kg, M=M, s=s)
    pci.applyTo(kmax=kmax, dk=dk, x_I_max=x_I_max, dx_I=dx_I)

    fig, axes = pci.plotTpp(interpolation='nearest')

    plt.show()
