import numpy as np
import matplotlib.pyplot as plt
from wavenumber_response import PhaseContrastImaging


# Optical parameters of PCI system
kg = 2.  # [kg] = 1 / w0;  kg = 2 is the diffraction limit
M = 0.5  # [M] = unitless; typical value...

# Computational grid
kmax = 5 * kg       # [kmax] = [kg]
dk = 0.01           # [dk] = [kg]

xmax = 1.5          # [xmax] = w0
dx = 0.01           # [dx] = w0
xmax_I = xmax * M   # [xmax_I] = M * w0
dx_I = dx * M       # [dx_I] = M * w0


if __name__ == '__main__':
    pci = PhaseContrastImaging(kg=kg, M=M, s=1)
    pci.applyTo(kmax=kmax, dk=dk, x_I_max=xmax_I, dx_I=dx_I)

    fig, axes = pci.plotResponse(interpolation='nearest')

    plt.show()
