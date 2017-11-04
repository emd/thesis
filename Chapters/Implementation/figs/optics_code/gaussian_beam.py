import numpy as np


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
            if (w is not None) and (R is not None):
                qinv = (1. / R) - (1j * wavelength / (np.pi * (w ** 2)))
                self.q = 1. / qinv
            else:
                raise ValueError(
                    'Both `w` & `R` must be specified if `q` is None')
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

    @property
    def w(self):
        'Get 1/e E radius.'
        qinv = 1. / self.q
        return np.sqrt(-self.wavelength / (np.pi * np.imag(qinv)))

    @property
    def z(self):
        'Get axial distance from beam waist, z; z > 0 => downstream of waist.'
        return np.real(self.q)

    @property
    def zR(self):
        'Get Rayleigh range.'
        return np.imag(self.q)

    @property
    def w0(self):
        'Get 1/e E radius at beam waist.'
        return np.sqrt(self.wavelength * self.zR / np.pi)


def w(z, w0, zR):
    '''Gaussian beam 1/e E radius.

    Parameters:
    -----------
    z - array_like, (`N`,)
        The axial distance from the beam waist.
        [z] = arbitrary units

    w0 - float
        1/e E radius of beam waist.
        [w0] = [z]

    zR - float
        Rayleigh range of beam.
        [zR] = [z]

    Returns:
    --------
    w - array_like, (`N`,)
        The beam 1/e E radius as a function of the axial distance `z`
        from the beam waist.
        [w] = [z]

    '''
    return w0 * np.sqrt(1 + ((z / zR) ** 2))


def R(z, zR):
    '''Gaussian beam radius of curvature.

    Parameters:
    -----------
    z - array_like, (`N`,)
        The axial distance from the beam waist.
        [z] = arbitrary units

    zR - float
        Rayleigh range of beam.
        [zR] = [z]

    Returns:
    --------
    R - array_like, (`N`,)
        The beam radius of curvature as a function of the axial distance `z`
        from the beam waist.
        [R] = [z]

    '''
    return z * (1 + ((zR / z) ** 2))
