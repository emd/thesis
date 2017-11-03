import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm

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
    90 * in2m, (110 * in2m) + dz, dz)
z_L1L2_choices = np.arange(             # [z_L1L2_choices] = m
    15 * in2m, (35 * in2m) + dz, dz)

# Plot parameters
figsize = (6, 7)
cmap_sequential = 'viridis'
cmap_diverging = 'RdBu'
cbar_orientation = 'vertical'
fontsize = 12

# Additional parameters
w0 = 1.25e-3  # 1/e E radius at laser source, [w0] = m
s = 1e-3      # detector side length, [s] = m


# Helper class for creating a diverging colormap with asymmetric limits;
# taken from Joe Kington at:
#
#   https://stackoverflow.com/a/20146989/5469497
#
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


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
    z = np.zeros(shape)
    zR = np.zeros(shape)

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

            # Propagate beam from source to detector
            gdet = gsource.applyABCD(ABCD * source_to_midplane_ABCD())

            # Determine plasma-arm radius of curvature at image plane and
            # corresponding maximum curvature-induced phase shift
            Rp = gdet.R
            delta = np.abs((1. / Rp) - (1. / Rr))
            dphi_kappa[i, j] = 0.25 * k0 * (s ** 2) * delta

            # Get axial distance to waist and Rayleigh range
            z[i, j] = gdet.z
            zR[i, j] = gdet.zR

    fig, axes = plt.subplots(
        3, 2, figsize=figsize, sharex=True, sharey=True,
        subplot_kw={'aspect': 1, 'adjustable': 'box-forced'})

    # 2nd imaging lens to detector
    level_spacing00 = 1.
    levels00 = np.arange(10, 20 + level_spacing00, level_spacing00)
    C00 = axes[0, 0].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, z_L2det.T / in2m,
        levels00, cmap=cmap_sequential)
    cb00 = plt.colorbar(C00, ax=axes[0, 0], orientation=cbar_orientation)
    cb00.set_ticks(levels00[::2])
    axes[0, 0].set_ylabel(
        r'$d_{\mathrm{L1, L2}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[0, 0].set_title(
        r'$d_{L2,\mathcal{I}} \, [\mathrm{in}]$',
        fontsize=fontsize)

    # Magnification
    level_spacing01 = 0.01
    levels01 = np.arange(0.06, 0.139 + level_spacing01, level_spacing01)
    C01 = axes[0, 1].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, M.T,
        levels01, cmap=cmap_sequential)
    cb01 = plt.colorbar(C01, ax=axes[0, 1], orientation=cbar_orientation)
    cb01.set_ticks(levels01[::2])
    axes[0, 1].set_title(
        r'$M \; [\mathrm{unitless}]$',
        fontsize=fontsize)

    # C of ABCD ray matrix
    level_spacing10 = 0.25
    levels10 = np.arange(-1.5, 0.5 + level_spacing10, level_spacing10)
    norm = MidpointNormalize(midpoint=0)
    C10 = axes[1, 0].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, C.T,
        levels10, cmap=cmap_diverging, norm=norm)
    cb10 = plt.colorbar(C10, ax=axes[1, 0], orientation=cbar_orientation)
    cb10.set_ticks(levels10[::2])
    axes[1, 0].set_ylabel(
        r'$d_{\mathrm{L1, L2}} \, [\mathrm{in}]$',
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
        cmap=cmap_sequential)
    cb11 = plt.colorbar(C11, ax=axes[1, 1], orientation=cbar_orientation)
    cb11.set_ticks(levels11[::2])
    axes[1, 1].set_title(
        r'$\mathrm{max}(\delta \phi_{\kappa}) \; [\mathrm{rad}]$',
        fontsize=fontsize)

    # Axial distance to beam waist, z
    levels20 = np.array([-10., -3., -1., -0.3, -0.1, 0.1, 0.3, 1., 3., 10.])
    linthresh = 0.01
    C20 = axes[2, 0].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, z.T,
        levels20,
        norm=SymLogNorm(linthresh, linscale=linthresh),
        cmap=cmap_diverging)
    cb20 = plt.colorbar(C20, ax=axes[2, 0], orientation=cbar_orientation)
    axes[2, 0].set_xlabel(
        r'$d_{\mathrm{P2, L1}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[2, 0].set_ylabel(
        r'$d_{\mathrm{L1, L2}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[2, 0].set_title(
        r'$z \; [\mathrm{m}]$',
        fontsize=fontsize)

    # Rayleigh range, zR
    level_spacing21 = 0.1
    levels21 = np.arange(level_spacing21, 0.99 + level_spacing21, level_spacing21)
    levels21 = np.array([0.03, 0.1, 0.3, 1., 3., 10., 30.])
    C21 = axes[2, 1].contourf(
        z_P2L1_choices / in2m, z_L1L2_choices / in2m, zR.T,
        levels21,
        norm=LogNorm(),
        cmap=cmap_sequential)
    cb21 = plt.colorbar(C21, ax=axes[2, 1], orientation=cbar_orientation)
    cb21.set_ticks(levels21[1::2])
    axes[2, 1].set_xlabel(
        r'$d_{\mathrm{P2, L1}} \, [\mathrm{in}]$',
        fontsize=fontsize)
    axes[2, 1].set_title(
        r'$z_R \; [\mathrm{m}]$',
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
