'''Measurements with calibrated microphone from

    /fusion/pillar-archive/u/davisem/HWlogbook/1161025.txt

and characterization of the sound waves from

    ./tymphany_measurements.py

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate as interp
from distinct_colours import get_distinct

from wavenumber import wavenumber


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


class PressureField(object):
    def __init__(self, f, z, x):
        '''Create an instance of `PressureField`.

        Input parameters:
        -----------------
        f - float
            The sound-wave frequency.
            [f] = kHz

        z - array_like, (`M`,)
            The axial distance from the speaker face.
            [z] = cm

        x - array_like, (`M`,)
            The transverse distance from the speaker's symmetry axis.
            [x] = cm

        '''
        self.f = f
        self.k = wavenumber(f)  # [self.k] = cm^{-1}
        self.xx, self.zz = np.meshgrid(x, z)

    def getAmplitude(self):
        '''Get single-sided amplitude (in Pa) of pressure fluctuation
        by interpolating on-axis amplitudes measured at various
        frequencies and on-axis heights.

        '''
        mV_per_Pa = 2.52  # microphone calibration factor

        # Measurement points
        f_meas = np.arange(1., 31., 1.)     # [f_meas] = kHz
        z_meas = np.array([2.5, 5.5, 8.5])  # [z_meas] = cm
        ff, zz = np.meshgrid(f_meas, z_meas)

        # Measured values
        # [Vpp_meas] = mV
        Vpp_meas = np.array([
            [35.0, 67.2, 43.2, 40.4, 40.0,
             44.0, 40.0, 36.0, 36.0, 33.6,
             35.2, 30.4, 36.0, 29.2, 28.8,
             32.0, 33.6, 38.4, 38.8, 35.6,
             40.8, 50.4, 56.8, 50.4, 40.8,
             42.0, 42.4, 40.0, 40.4, 36.4],
            [17.2, 40.0, 28.0, 23.2, 16.8,
             23.2, 22.8, 23.2, 20.4, 22.4,
             20.0, 18.8, 23.2, 20.8, 22.8,
             19.2, 21.2, 22.0, 25.6, 24.0,
             21.6, 22.0, 20.0, 27.2, 27.2,
             28.4, 28.8, 27.6, 28.8, 31.2],
            [10.0, 26.0, 18.0, 16.4, 12.8,
             15.6, 14.8, 16.4, 16.4, 16.8,
             14.8, 12.8, 14.8, 16.0, 16.0,
             14.4, 16.0, 18.0, 18.0, 18.0,
             15.2, 14.0, 14.8, 16.4, 16.8,
             17.2, 19.6, 21.6, 21.2, 20.8]])

        Vpp_Rbf = interp.Rbf(ff, zz, Vpp_meas)

        Vpp = Vpp_Rbf(
            self.f * np.ones(self.zz.shape),
            self.zz)

        p0 = 0.5 * Vpp / mV_per_Pa

        return p0

    def getGaussianEnvelope(self, w0=4.):
        '''To good approximation (especially for larger frequencies),
        the amplitude envelope is Gaussian.

        '''
        # Measurement points
        f_meas = np.arange(5., 26., 5.)     # [f_meas] = kHz
        z_meas = np.array([2.5, 5.5, 8.5])  # [z_meas] = cm
        ff, zz = np.meshgrid(f_meas, z_meas)

        # Fitted Gaussian widths, from ./tymphany_measurements.py
        #
        # Also, the f = 25 kHz width at a height of 2.5 cm was
        # not very trustworthy, so I modified this point to obey
        # the qualitative & quantitative trends of the other points.
        #
        # [w_fit] = cm
        w_fit = np.array([
            [6.03713763, 3.2874123, 2.897124, 1.98873143, 0.91667516],
            [12.12209224, 5.31404444, 4.04580896, 3.0623345 , 1.49213542],
            [16.5023127, 6.82752067, 6.01321625, 4.23824505, 2.77332402]])

        w_Rbf = interp.Rbf(ff, zz, w_fit)

        w = w_Rbf(
            self.f * np.ones(self.zz.shape),
            self.zz)

        return np.exp(-((self.xx / w) ** 2))

    def getPhase(self):
        '''To good approximation, the sound waves are spherical.
        Get the corresponding phase.

        '''
        return self.k * np.sqrt((self.xx ** 2) + (self.zz ** 2))

    def getPressureField(self):
        p0 = self.getAmplitude()
        envelope = self.getGaussianEnvelope()
        theta = self.getPhase()

        # Note that above `theta` corresponds to
        # a "snapshot" of the wave at a single point
        # in time; however, we want the RMS value, so
        # we should compute the wave structure
        # throughout a full 2 * pi cycle.
        full_cycle = np.arange(0, 2 * np.pi, np.pi / 180)
        theta = theta[..., np.newaxis] + full_cycle

        p0 = p0[..., np.newaxis]
        envelope = envelope[..., np.newaxis]

        return p0 * envelope * np.cos(theta)


def plot_example_pressure_field(f=15., cmap='RdBu', fontsize=12):
    # Height above speaker face
    # [z] = cm
    zmin = 2.5
    zmax = 8.5
    dz = 0.1
    z = np.arange(zmin, zmax + dz, dz)

    # Radial distance from symmetry axis
    # [x] = cm
    xmin = -5.
    xmax = 5.
    dx = 0.1
    x = np.arange(xmin, xmax + dx, dx)

    P = PressureField(f, z, x)

    M = 0.6
    figsize = (M * (xmax - xmin), M * (zmax - zmin))
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    levels = np.arange(-6., 7., 1.)
    C = ax.contourf(
        P.xx,
        P.zz,
        P.getPressureField()[..., 0],
        levels,
        cmap=cmap)
    cb = plt.colorbar(
        C,
        ax=ax,
        shrink=0.825)
    ax.set_aspect('equal')

    ax.set_xlabel(
        r'$\mathregular{\rho \; [cm]}$',
        fontsize=fontsize)
    ax.set_ylabel(
        r'$\mathregular{z \; [cm]}$',
        fontsize=fontsize)
    ax.set_title(
        r'$\mathregular{\widetilde{p} \; [Pa] \; @ \; f = %i \, kHz}$' % np.int(f),
        fontsize=fontsize)
    ax.set_xlim([xmin, xmax])

    plt.tight_layout()
    plt.show()

    return


def plot_phase_response():
    zmin = 2.5
    zmax = 8.5
    dz = 0.25
    z = np.arange(zmin, zmax + dz, dz)

    xmin = -10.
    xmax = 10.
    dx = 0.1
    x = np.arange(xmin, xmax + dx, dx)

    # From Chris' code
    dNdp = 7.86e-7 / (1.4 * 300)

    norm = MidpointNormalize(midpoint=0)

    # f = 10.
    freqs = np.arange(2.5, 25., 1)

    varphi = np.zeros((len(freqs), len(z)))
    for find, f in enumerate(freqs):
        # Compute pressure field and change to index of refraction
        P = PressureField(f, z, x)
        dN = dNdp * P.getPressureField()

        # plt.figure()
        # plt.contourf(P.xx, P.zz, dN[..., 0], norm=norm, cmap='RdBu')
        # plt.gca().set_aspect('equal')
        # plt.colorbar()
        # plt.show()

        # Integrate along beam path
        lambda0 = 10.6e-6 * 1e2   # [lambda0] = cm
        k0 = 2 * np.pi / lambda0  # [k0] = cm^{-1}
        phi = k0 * np.sum(dN, axis=1) * dx

        varphi[find, :] = np.var(phi, axis=-1)

        # plt.figure()
        # plt.plot(z, np.var(phi, axis=-1))
        # plt.show()

    # plt.figure()
    # plt.contourf(
    #     wavenumber(freqs), z, varphi.T,
    #     norm=LogNorm(vmin=1e-9, vmax=1e-7),
    #     cmap='viridis')
    # plt.colorbar()
    # plt.show()

    plt.figure()

    k = wavenumber(freqs)
    w = (np.sinc(k / 5.)) ** 2

    # for zind in np.arange(0, len(z), 1):
    #     plt.semilogy(
    #         k,
    #         (w * varphi[:, zind]) + 2e-9)

    ymin = w * np.min(varphi, axis=-1)
    ymax = w * np.max(varphi, axis=-1)

    ymin += 2e-9
    ymax += 2e-9

    plt.fill_between(k, ymin, ymax)
    plt.yscale('log')

    plt.xlabel(
        r'$\mathregular{k \; [cm^{-1}]}$',
        fontsize=16)
    plt.ylabel(
        r'$\mathregular{\widetilde{\phi}^2 \; [rad^{2}]}$',
        fontsize=16)
    # plt.legend(loc='best')
    plt.show()

    return


if __name__ == '__main__':
    plot_example_pressure_field()
    # plot_phase_response()
