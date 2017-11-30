'''Measurements with calibrated microphone from

    /fusion/pillar-archive/u/davisem/HWlogbook/1161025.txt

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy.optimize import curve_fit
from distinct_colours import get_distinct

from wavenumber import wavenumber


# Plotting parameters
fontsize = 12

# Sound speed
cs = 343e-4  # [cs] = cm / (micro-s)


def time_lag_expected(dist, x0):
    '''Expected time delay between wave at (x, H) relative to (x0, H),
    where x0 is the on-axis position.

    Input parameters:
    -----------------
    dist - array_like, (`1 + N`,)
        The first term gives the on-axis height, and
        the remainder of the terms give the transverse positions,
        which are not necessarily centered on axis.
        [dist] = cm

    x0 - float
        The on-axis position.
        [x0] = cm

    Returns:
    --------
    tau - array_like, (`N`,)
        The expected time delay at each point (x, H).
        [tau] = micro-s

    '''
    H = dist[0]
    x = dist[1:]

    term1 = np.sqrt((H ** 2) + ((x - x0) ** 2))
    term2 = H
    tau = (term1 - term2) / cs

    return tau


def gaussian(x, w, a):
    return a * np.exp(-((x / w) ** 2))


class SpatialProfile(object):
    def __init__(self, H, f, V, tau, dx=1.):
        '''Create instance of `SpatialProfile` class.

        Input parameters:
        -----------------
        H - float
            Height above speaker.
            [H] = cm

        f - float
            Frequency.
            [f] = kHz

        V - array_like, (`N`,)
            The wave amplitude at each spatial position.
            [V] = arbitrary

        tau - array_like, (`N`,)
            The time lag between the measured wave at
            each spatial location vs. a reference wave.
            [tau] = micro-s

        dx - float
            The spacing between adjacent measurement locations.
            [dx] = cm

        '''
        self.H = H
        self.f = f
        self.V = V.copy()
        self.tau = tau - np.min(tau)

        # Determine on-axis position and create coordinate system
        # centered on-axis
        self._x = dx * np.arange(len(self.V))
        self.x0 = self.getCenter()
        self.x = self._x - self.x0

        # Determine width and amplitude of fitted Gaussian
        self.w, self.a = self.getGaussianFit()

    def getCenter(self):
        dist = np.concatenate(([self.H], self._x))
        popt, pcov = curve_fit(time_lag_expected, dist, self.tau)
        x0 = popt[0]
        return x0

    def getGaussianFit(self):
        popt, pcov = curve_fit(gaussian, self.x, self.V)

        # A negative width is equivalent to a positive width, as
        # it only comes into the formula as a squared value;
        # ensure we are dealing with a positive width
        w = np.abs(popt[0])

        a = popt[1]

        return w, a

    def plotProfile(self):
        fig, axes = plt.subplots(2, 1, sharex=True)

        axes[0].plot(self.x, self.V)
        axes[0].set_ylabel(r'$\mathregular{V_{mic} \; [V_{pp}]}$')

        axes[1].plot(self.x, self.tau)
        axes[1].set_ylabel(r'$\mathregular{dt \; [\mu s]}$')

        plt.suptitle('H = %.1f cm, f = %.1f kHz' % (self.H, self.f))
        plt.show()

        return


def voltage_profile(H, f):
    if H == 8.5:
        if f == 5.:
            return np.array([
                11.0, 10.0, 9.8, 9.8,
                9.8, 9.8, 9.8, 10.0,
                10.8, 10.9, 10.3, 10.0,
                9.6, 9.4, 8.6, 8.2,
                7.3, 5.7, 4.3, 3.3])
        elif f == 10.:
            return np.array([
                7.5, 9.6, 11.6, 13.2,
                14.9, 15.7, 15.3, 14.1,
                12.0, 10.2, 8.4, 6.5,
                5.9, 5.1, 3.8, 3.3])
        elif f == 15.:
            return np.array([
                5.8, 7.3, 9.6, 11.3,
                12.8, 13.5, 12.9, 11.2,
                9.9, 8.2, 6.2, 4.8,
                3.7, 3.2, 3.2])
        elif f == 20.:
            return np.array([
                3.3, 5.7, 9.4, 13.7,
                16.1, 16.6, 15.3, 13.1,
                10.4, 6.9, 5.0, 3.2,
                1.7])
        elif f == 25.:
            return np.array([
                1.9, 4.1, 9.4, 14.4,
                15.3, 14.2, 10.1, 5.3,
                1.9])
        elif f == 30.:
            return np.array([
                3.4, 4.6, 4.3, 2.4,
                7.5, 15.6, 21.6, 20.1,
                11.5, 4.6, 3.0, 4.5,
                5.5, 4.1])
        else:
            raise ValueError('Frequency %.1f not measured' % f)
    elif H == 5.5:
        if f == 5.:
            return np.array([
                15.8, 15.9, 16.3, 16.5,
                15.9, 15.7, 14.9, 15.0,
                15.6, 16.1, 15.3, 14.3,
                13.1, 11.8, 10.9,  8.9,
                6.5, 4.8, 4.5, 4.9,
                4.9, 3.1, 3.4, 3.1])
        elif f == 10.:
            return np.array([
                6.7, 9.1, 13.6, 16.9,
                19.4, 20.4, 19.7, 17.3,
                14.2, 10.9, 7.9, 6.2,
                5.1, 4.9, 3.7])
        elif f == 15.:
            return np.array([
                3.6, 5.8, 8.9, 12.3,
                16.5, 19.4, 17.8, 14.0,
                9.4, 6.5, 4.8, 3.4,
                3.8, 3.0])
        elif f == 20.:
            return np.array([
                2.8, 1.6, 5.9, 14.7,
                20.3, 21.9, 19.9, 15.4,
                9.7, 5.1, 2.5, 2.7])
        elif f == 25.:
            return np.array([
                # 6.3, 7.3, 5.3,
                1.1,
                14.3, 23.7, 19.5, 5.1,
                # 2.0, 4.5, 4.6
                ])
        elif f == 30.:
            return np.array([
                5.3, 16.4, 24.2, 20.2,
                5.6, 0., 4.3, 4.4])
        else:
            raise ValueError('Frequency %.1f not measured' % f)
    elif H == 2.5:
        if f == 5.:
            return np.array([
                15.0, 20.5, 23.4, 26.1,
                32.3, 36.2, 33.5, 27.6,
                23.7, 21.3, 18.1, 13.1,
                9.9,  8.7])
        elif f == 10.:
            return np.array([
                5.3, 6.5, 10.1, 17.7,
                27.6, 32.1, 29.5, 22.1,
                13.7, 7.8, 6.2, 4.7])
        elif f == 15.:
            return np.array([
                6.5, 11.0, 22.4, 27.2,
                23.3, 15.0, 9.8, 7.7,
                5.0])
        elif f == 20.:
            return np.array([
                5.9, 24.4, 35.2, 31.9,
                13.0, 6.1, 5.1])
        elif f == 25.:
            return np.array([
                5.3, 33.3, 12.0])
        else:
            raise ValueError('Frequency %.1f not measured' % f)
    else:
        raise ValueError('Height %.1f not measured' % H)


def lag_profile(H, f):
    if H == 8.5:
        if f == 5.:
            return np.array([
                -5., -16., -26., -30.,
                -32., -28., -18., -12.,
                -4., 8., 20., 35.,
                50., 68., 86., 104.,
                125., 148., 170., 201.])
        elif f == 10.:
            return np.array([
                54., 45., 35., 26.,
                21., 20., 26., 26.,
                33., 43., 58., 72.,
                88., 103., 124., 144.])
        elif f == 15.:
            return np.array([
                -3., -13., -22.,-29.,
                -35., -36., -30., -25.,
                -17., -7., 4., 13.,
                30., 46., 63])
        elif f == 20.:
            return np.array([
                25., 15., 5., -2.,
                -5., -8., -5., -1.,
                6., 16., 26., 40.,
                55.])
        elif f == 25.:
            return np.array([
                -7., -14., -24., -30.,
                -31., -28., -26., -16.,
                -6.])
        elif f == 30.:
            return np.array([
                19., 6., -6., -10.,
                -10., -15., -16., -15.,
                -9., -4., -4., 5.,
                20., 8.])
        else:
            raise ValueError('Frequency %.1f not measured' % f)
    elif H == 5.5:
        if f == 5.:
            return np.array([
                132., 113.,  96.,  83.,
                78.,  75.,  79.,  88.,
                102., 115., 130., 147.,
                163., 185., 203., 225.,
                248., 273., 308., 333.,
                361., 387., 418., 441.])
        elif f == 10.:
            return np.array([
                80., 62., 49., 41.,
                37., 34., 38., 43.,
                50., 62., 75., 99.,
                120., 140., 157.])
        elif f == 15.:
            return np.array([
                58., 44., 31., 22.,
                17., 15., 19., 20.,
                31., 44., 55., 75.,
                92., 109])
        elif f == 20.:
            return np.array([
                34., 30., 23., 13.,
                7., 6., 8., 13.,
                22., 33., 45., 56.])
        elif f == 25.:
            return np.array([
                # 35., 20., 2.,
                19.,
                3., 1., 4., 11.,
                # 2., 16., 32.
                ])
        elif f == 30.:
            return np.array([
                10., 2., -1., 1.,
                9., np.nan, 17., 35.])
        else:
            raise ValueError('Frequency %.1f not measured' % f)
    elif H == 2.5:
        if f == 5.:
            return np.array([
                67., 45., 31., 12.,
                -2., -8., -2., 10.,
                27., 45., 64., 90.,
                115., 143.])
        elif f == 10.:
            return np.array([
                21., 2., -23., -37.,
                -45., -49., -45., -37.,
                -25., -6., 16., 38.])
        elif f == 15.:
            return np.array([
                19.,  6.,  1., -1.,
                1.,  7., 19., 33.,
                57.])
        elif f == 20.:
            return np.array([
                26., 26., 25., 26.,
                30., 37., 50])
        elif f == 25.:
            return np.array([
                30., 2., 1.])
        else:
            raise ValueError('Frequency %.1f not measured' % f)
    else:
        raise ValueError('Height %.1f not measured' % H)


def plot_on_axis_amplitude():
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

    cols = get_distinct(len(z_meas))
    syms = ['s', 'o', '^']

    k_meas = wavenumber(f_meas)

    plt.figure()
    for zind, z in enumerate(z_meas):
        plt.plot(
            k_meas,
            Vpp_meas[zind, :],
            '-%s' % syms[zind],
            c=cols[zind],
            label='z = %.1f cm' % z)

    plt.xlabel(
        r'$\mathregular{k \; [cm^{-1}]}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{V_{mic} \; [mV_{pp}]}$',
        fontsize=fontsize)
    plt.legend(loc='best')

    plt.show()

    return


def plot_spatial_envelope(z=5.5, fmin=5., fmax=25., df=5.):
    # Measured frequencies
    # [f_meas] = kHz
    f_meas = np.arange(fmin, fmax + df, df)

    cols = get_distinct(len(f_meas))
    syms = ['s', 'o', '^', 'v', 'd']

    # Finely spaced computational grid for evaluating fit
    x = np.arange(-6, 18, 0.1)

    plt.figure()
    for find, f in enumerate(f_meas):
        prof = SpatialProfile(
            z, f,
            voltage_profile(z, f),
            lag_profile(z, f))

        # Plot data points
        plt.plot(
            prof.x,
            prof.V,
            '%s' % syms[find],
            c=cols[find],
            label='f = %.1f kHz' % f)

        # Plot fit
        plt.plot(
            x,
            gaussian(x, prof.w, prof.a),
            c=cols[find])

    plt.xlabel(
        r'$\mathregular{\rho \; [cm]}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{V_{mic} \; [mV_{pp}]}$',
        fontsize=fontsize)
    plt.title(r'$\mathregular{z = %.1f cm}$' % z)
    plt.legend(loc='best')

    plt.show()

    return


def plot_gaussian_widths(fmin=5., fmax=25., df=5.):
    # Measurement points
    f_meas = np.arange(fmin, fmax + df, df)  # [f_meas] = kHz
    z_meas = np.array([2.5, 5.5, 8.5])       # [z_meas] = cm

    w = np.zeros((len(z_meas), len(f_meas)))

    for zind, z in enumerate(z_meas):
        for find, f in enumerate(f_meas):
            prof = SpatialProfile(
                z, f,
                voltage_profile(z, f),
                lag_profile(z, f))

            w[zind, find] = prof.w

    k_meas = wavenumber(f_meas)

    cols = get_distinct(len(z_meas))
    syms = ['s', 'o', '^']

    plt.figure()
    for zind, z in enumerate(z_meas):
        if z == 2.5:
            kind = slice(None, -1)
        else:
            kind = slice(None, None)

        plt.plot(
            k_meas[kind],
            w[zind, :][kind],
            '-%s' % syms[zind],
            c=cols[zind],
            label='z = %.1f cm' % z)

    plt.xlabel(r'$\mathregular{k \; [cm^{-1}]}$', fontsize=fontsize)
    plt.ylabel(r'$\mathregular{w \; [cm]}$', fontsize=fontsize)
    plt.legend(loc='best')

    plt.show()

    return


def plot_wavefront_phasing(fmin=5., fmax=25., df=5.):
    # Measurement points
    f_meas = np.arange(fmin, fmax + df, df)  # [f_meas] = kHz
    z_meas = np.array([2.5, 5.5, 8.5])       # [z_meas] = cm

    cols = get_distinct(len(z_meas))
    syms = ['s', 'o', '^', 'v', 'd']

    # Finely spaced computational grid for evaluating fit
    x = np.arange(-6, 18, 0.1)

    plt.figure()
    for zind, z in enumerate(z_meas):
        for find, f in enumerate(f_meas):
            prof = SpatialProfile(
                z, f,
                voltage_profile(z, f),
                lag_profile(z, f))

            # Plot raw data points
            plt.plot(
                prof.x,
                prof.tau,
                '%s' % syms[find],
                c=cols[zind])

        # Plot expected trace
        dist = np.concatenate(([prof.H], x))
        plt.plot(
            x,
            time_lag_expected(dist, 0),
            c=cols[zind],
            label='z = %.1f cm' % z)

    plt.xlabel(
        r'$\mathregular{\rho \; [cm]}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{\tau \; [\mu s]}$',
        fontsize=fontsize)
    plt.legend(loc='best')

    plt.show()

    return


if __name__ == '__main__':
    plot_on_axis_amplitude()

    plot_wavefront_phasing()

    plot_spatial_envelope(z=2.5, fmin=5., fmax=25., df=5.)
    plot_spatial_envelope(z=5.5, fmin=5., fmax=25., df=5.)
    plot_spatial_envelope(z=8.5, fmin=5., fmax=25., df=5.)

    plot_gaussian_widths()
