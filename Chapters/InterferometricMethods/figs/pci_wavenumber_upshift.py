import matplotlib.pyplot as plt
from wavenumber_response import PhaseContrastImaging


# Optical parameters of PCI system
kg = 2.  # kg = 2 / w0 is the diffraction limit
M = 0.5  # typical...

# Computational grid
kmax = 5 * kg
dk = 0.1
x_I_max = 2.
dx_I = 0.1


if __name__ == '__main__':
    pci = PhaseContrastImaging(kg=kg, M=M)
    pci.applyTo(kmax=kmax, dk=dk, x_I_max=x_I_max, dx_I=dx_I)

    ax = pci.plotMeasuredWavenumbers(interpolation='nearest')

    plt.show()
