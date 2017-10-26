import numpy as np
import matplotlib.pyplot as plt

from ABCD_matrices import prop, lens
from geometric_optics import image_distance
from gaussian_beam import GaussianBeam

from expansion_optics import in2m, f_P1, source_to_midplane_ABCD
from imaging_optics_interferometer import (
    f_P2, f_L1, f_L2, z_midP2, z_ref)
from imaging_optics_interferometer import z_P2L1 as z_P2L1_selected
from imaging_optics_interferometer import z_L1L2 as z_L1L2_selected


# Design parameters to be scanned
dz = 0.25 * in2m                        # [dz] = m
z_P2L1_choices = np.arange(             # [z_P2L1_choices] = m
    90 * in2m, (120 * in2m) + dz, dz)
z_L1L2_choices = np.arange(             # [z_L1L2_choices] = m
    15 * in2m, (30 * in2m) + dz, dz)

# Plot parameters
cmap = 'viridis'
cbar_orientation = 'vertical'
fontsize = 12

# Additional parameters
w0 = 1.25e-3  # 1/e E radius at laser source, [w0] = m
s = 1e-3      # detector side length, [s] = m


if __name__ == '__main__':
    # Gaussian beam at source
    gsource = GaussianBeam(None, w=w0, R=np.inf)
    k0 = 2 * np.pi / gsource.wavelength  # wavenumber, [k0] = m^{-1}

    # Reference arm radius of curvature @ detector
    Rr = (gsource.applyABCD(prop(z_ref))).R

    # Initialize design arrays
    shape = (len(z_P2L1_choices), len(z_L1L2_choices))
    z_L2det = np.zeros(shape)
    M = np.zeros(shape)
    C = np.zeros(shape)
    dphi_kappa = np.zeros(shape)

    for i, z_P2L1 in enumerate(z_P2L1_choices):
        for j, z_L1L2 in enumerate(z_L1L2_choices):
            # Propagate from tokamak midplane up to and through
            # the final imaging lens (L2), ensuring to multiply
            # by new matrix elements *on the left*
            ABCD = prop(z_midP2)
            ABCD = lens(f_P2) * ABCD
            ABCD = prop(z_P2L1) * ABCD
            ABCD = lens(f_L1) * ABCD
            ABCD = prop(z_L1L2) * ABCD
            ABCD = lens(f_L2) * ABCD

            # Determine distance from L2 to detector
            z_L2det[i, j] = image_distance(ABCD)

            # Determine ABCD matrix for full imaging system
            ABCD = prop(z_L2det[i, j]) * ABCD

            # Extract magnification M and ray-matrix element C
            M[i, j] = ABCD[0, 0]
            C[i, j] = ABCD[1, 0]

            # Determine plasma-arm radius of curvature at image plane
            Rp = (gsource.applyABCD(ABCD * source_to_midplane_ABCD())).R

            delta = np.abs((1. / Rp) - (1. / Rr))
            dphi_kappa[i, j] = 0.25 * k0 * (s ** 2) * delta

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    # 2nd imaging lens to detector
    level_spacing00 = 1.
    levels00 = np.arange(10, 20 + level_spacing00, level_spacing00)
    C00 = axes[0, 0].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, z_L2det.T / in2m,
        levels00, cmap=cmap)
    cb00 = plt.colorbar(C00, ax=axes[0, 0], orientation=cbar_orientation)
    cb00.set_ticks(levels00[::2])
    axes[0, 0].set_ylabel(
        r'$z_{\mathrm{L1, L2}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[0, 0].set_title(
        r'$z_{L2,\mathrm{det}} \, [\mathrm{in}]$',
        fontsize=fontsize)

    # Magnification
    level_spacing01 = 0.01
    levels01 = np.arange(0.06, 0.139 + level_spacing01, level_spacing01)
    C01 = axes[0, 1].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, M.T,
        levels01, cmap=cmap)
    cb01 = plt.colorbar(C01, ax=axes[0, 1], orientation=cbar_orientation)
    cb01.set_ticks(levels01[::2])
    axes[0, 1].set_title(
        r'$M \; [\mathrm{unitless}]$',
        fontsize=fontsize)

    # C of ABCD ray matrix
    level_spacing10 = 0.25
    levels10 = np.arange(-1.5, 0.5 + level_spacing10, level_spacing10)
    C10 = axes[1, 0].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, C.T,
        levels10, cmap=cmap)
    cb10 = plt.colorbar(C10, ax=axes[1, 0], orientation=cbar_orientation)
    cb10.set_ticks(levels10[::2])
    axes[1, 0].set_xlabel(
        r'$z_{\mathrm{P2, L1}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[1, 0].set_ylabel(
        r'$z_{\mathrm{L1, L2}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[1, 0].set_title(
        r'$C \; [\mathrm{m}^{-1}]$',
        fontsize=fontsize)

    # dphi_kappa
    level_spacing11 = 0.1
    levels11 = np.arange(0, 0.99 + level_spacing11, level_spacing11)
    C11 = axes[1, 1].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, dphi_kappa.T,
        levels11,
        cmap=cmap)
    cb11 = plt.colorbar(C11, ax=axes[1, 1], orientation=cbar_orientation)
    cb10.set_ticks(levels10[::2])
    axes[1, 1].set_xlabel(
        r'$z_{\mathrm{P2, L1}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[1, 1].set_title(
        r'$\mathrm{max}(\delta \phi_{\kappa}) \; [\mathrm{rad}]$',
        fontsize=fontsize)

    # Add selected design point to each contour plot
    for i in np.arange(axes.shape[0]):
        for j in np.arange(axes.shape[1]):
            axes[i, j].plot(
                z_P2L1_selected / in2m,
                z_L1L2_selected / in2m,
                'D', c='darkred')

    plt.tight_layout()

    plt.show()