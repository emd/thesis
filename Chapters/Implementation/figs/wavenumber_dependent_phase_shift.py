import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


in2m = 0.0254

# System magnification
M = 0.08

# Probe beam
lambda0 = 10.6e-6           # [lambda0] = m
k0 = 2 * np.pi / lambda0    # [k0] = m^{-1}

# Wavenumber computational grid
dk = 10     # [dk] = m^{-1}
kmax = 5e2  # [kmax] = m^{-1}

# Misalignment computational grid
ddz = 0.01    # [ddz] = in
dz_max = 1.   # [dz_max] = in

fontsize = 12
figsize = (5, 4)


if __name__ == '__main__':
    # 1d computational grid
    dz = np.arange(ddz, dz_max + ddz, ddz) * in2m  # [dz] = m
    k = np.arange(dk, kmax + dk, dk)                # [k] = m

    # 2d computational grid
    dzdz, kk = np.meshgrid(dz, k)

    # Wavenumber-dependent phase shift
    mu = ((kk ** 2) / (2 * (M ** 2) * k0)) * dzdz

    plt.figure(figsize=figsize)
    levels = np.logspace(-5, 0, 11)
    C = plt.contourf(
        dzdz / in2m, kk * 1e-2, mu,
        levels,
        norm=LogNorm())
    plt.colorbar(C)
    plt.xlabel(
        r'$|\delta z_{\mathcal{I}}| \; [\mathrm{in}]$',
        fontsize=fontsize)
    plt.ylabel(
        r'$|k| \; [\mathrm{cm}^{-1}]$',
        fontsize=fontsize)
    plt.title(r'$|\mu| \; [\mathrm{rad}]$')
    plt.tight_layout()
    plt.show()
