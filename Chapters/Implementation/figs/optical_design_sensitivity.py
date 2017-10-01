import numpy as np
import matplotlib.pyplot as plt


in2m = 0.0254  # i.e. z[m] = z[in] * in2m

# "Independent" design parameters
dz = 0.25 * in2m                        # [dz] = m
z_P2L1_choices = np.arange(             # [z_P2L1_choices] = m
    90 * in2m, (120 * in2m) + dz, dz)
z_L1L2_choices = np.arange(             # [z_L1L2_choices] = m
    15 * in2m, (30 * in2m) + dz, dz)

# Selected design parameters
z_P2L1_selected = 94.6 * in2m       # [z_P2L1_selected] = m
z_L1L2_selected = 23.875 * in2m     # [z_L1L2_selected] = m

# Plot parameters
cmap = 'viridis'
cbar_orientation = 'vertical'
fontsize = 12

# Focal lengths of focusing optics
f_exp = 10. * in2m  # [f_exp] = m
f_P1 = 80.7 * in2m  # [f_P1] = m
f_P2 = f_P1         # [f_P2] = m
f_L1 = 7.5 * in2m   # [f_L1] = m
f_L2 = 7.5 * in2m   # [f_L2] = m

# Fixed distances
z_P1mid = 353.6 * in2m  # 1st parabolic mirror to midplane, [z_P1mid] = m
z_midP2 = 263.8 * in2m  # midplane to 2nd parabolic mirror, [z_midP2] = m
z_ref = 59.375 * in2m   # total reference-arm distance, [z_ref] = m

# Additional parameters
w0 = 1.25e-3  # 1/e E radius at laser source, [w0] = m
s = 1e-3      # detector side length, [s] = m


class GaussianBeam(object):
    def __init__(self, q, w=None, R=None, wavelength=10.6e-6):
        '''Create an instance of `GaussianBeam` class.

        Input parameters:
        -----------------
        q - complex, or None
            Complex beam parameter. If `None`, use `w` and `R` to
            determine the corresponding value of `q`.
            [q] = m

        w - float
            1/e E radius of Gaussian beam. Only used if `q` is `None`.
            [w] = m

        R - float
            Radius of curvature of Gaussian beam. Only used if `q` is `None`.
            Note that a value of infinity corresponds to a beam waist.
            [R] = m

        wavelength - float
            Wavelength of Gaussian beam.
            Default of 10.6 microns corresponds to a CO2 laser.
            [wavelength] = m

        '''
        self.wavelength = wavelength

        if q is None:
            qinv = (1. / R) - (1j * wavelength / (np.pi * (w ** 2)))
            self.q = 1. / qinv
        else:
            self.q = q

    def applyABCD(self, ABCD):
        'Apply `ABCD` ray-matrix transformation to Gaussian beam.'
        A = ABCD[0, 0]
        B = ABCD[0, 1]
        C = ABCD[1, 0]
        D = ABCD[1, 1]

        num = (A * self.q) + B
        den = (C * self.q) + D

        q = num / den

        return GaussianBeam(q, wavelength=self.wavelength)

    @property
    def R(self):
        'Get radius of curvature.'
        Rinv = np.real(1. / self.q)

        if Rinv == 0:
            return np.inf
        else:
            return 1. / Rinv


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


def source_to_midplane_ABCD():
    'Get ABCD ray matrix from laser source to tokamak midplane.'
    # source to expansion lens
    ABCD = prop(68.4 * in2m)
    ABCD = lens(f_exp) * ABCD

    # Used in previous design work and should be "as built"...
    # beam approximately has waist after propagating this distance
    # and striking P1. Ideally, we'd want the waist to occur
    # at the tokamak midplane, but the Rayleigh length is so large
    # (~340 m) relative to the path length (~10 m) that it hardly
    # matters for our purposes *exactly* where the waist occurs.
    ABCD = prop(2.34333) * ABCD

    # Collimate
    ABCD = lens(f_P1) * ABCD

    # Propagate to midplane
    ABCD = prop(z_P1mid) * ABCD

    return ABCD


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
    # levels00 = np.arange(5, 20 + level_spacing00, level_spacing00)
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
