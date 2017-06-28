import numpy as np
import matplotlib.pyplot as plt


# "Independent" design parameters
dz = 1.00                                     # [dz] = inches
z_P2L1_choices = np.arange(90, 120 + dz, dz)  # [z_P2L1_choices] = inches
z_L1L2_choices = np.arange(14, 26 + dz, dz)   # [z_L1L2_choices] = inches

# Plot parameters
cmap = 'viridis'
cbar_orientation = 'vertical'
fontsize = 16

# Focal lengths of focusing optics
f_P2 = 80.7  # [f_P2 = inches
f_L1 = 7.5   # [f_L1] = inches
f_L2 = 7.5   # [f_L2] = inches

# Fixed distances
z_midP2 = 263.8  # midplane to 2nd parabolic mirror, [z_midP2] = inches


def lens(f):
    'Get ABCD matrix for lens of focal length `f`.'
    return np.matrix([
        [ 1.,     0.],
        [-1. / f, 1.]])


def prop(d):
    'Get ABCD matrix for constant-N propagation by distance `d`.'
    return np.matrix([
        [1., np.float(d)],
        [0.,          1.]])


def image_distance(ABCD):
    'Get image distance for optical system with ray matrix `ABCD`.'
    B = ABCD[0, 1]
    D = ABCD[1, 1]
    return -B / D


def test():
    # Focal length and object distances
    f = 1
    s = [2, 0.5]

    # Expected image distances
    sprime = [2, -1]

    for i in np.arange(len(s)):
        if image_distance(lens(f) * prop(s[i])) != sprime[i]:
            raise StandardError('Script *not* working correctly')

    print '\nTest cases passed'

    return


if __name__ == '__main__':
    test()

    # Initialize design arrays
    shape = (len(z_P2L1_choices), len(z_L1L2_choices))
    z_L2det = np.zeros(shape)
    M = np.zeros(shape)
    C = np.zeros(shape)
    dkappa = np.zeros(shape)

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

    # Convert C from in^{-1} to m^{-1}
    in2m = 0.0254  # i.e. z[m] = z[in] * in2m
    C /= in2m

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    # 2nd imaging lens to detector
    level_spacing00 = 1.
    # levels00 = np.arange(5, 20 + level_spacing00, level_spacing00)
    levels00 = np.arange(10, 20 + level_spacing00, level_spacing00)
    C00 = axes[0, 0].contourf(
        z_P2L1_choices, z_L1L2_choices, z_L2det.T,
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
        z_P2L1_choices, z_L1L2_choices, M.T,
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
        z_P2L1_choices, z_L1L2_choices, C.T,
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

    # dkappa
    # levels11 = 

    plt.show()
