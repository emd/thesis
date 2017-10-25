from nose import tools
import numpy as np

from gaussian_beam import GaussianBeam


def test_GaussianBeam__init__():
    # 1/e E waist of unity; with wavelength of `pi`, this should
    # result in a Rayleigh range of unity
    w = 1.
    R = np.inf
    wavelength = np.pi
    g = GaussianBeam(None, w=w, R=R, wavelength=wavelength)

    tools.assert_almost_equal(g.q, 0 + 1j)
    tools.assert_almost_equal(g.R, np.inf)
    tools.assert_almost_equal(g.w, w)
    tools.assert_almost_equal(g.z, 0)
    tools.assert_almost_equal(g.zR, 1)

    # 1/e E waist of 2; with wavelength of `pi`, this should
    # result in a Rayleigh range of 4
    w = 2.
    R = np.inf
    wavelength = np.pi
    g = GaussianBeam(None, w=w, R=R, wavelength=wavelength)

    tools.assert_almost_equal(g.q, 0 + 4j)
    tools.assert_almost_equal(g.R, np.inf)
    tools.assert_almost_equal(g.w, w)
    tools.assert_almost_equal(g.z, 0)
    tools.assert_almost_equal(g.zR, 4)

    # 1/e E waist of unity; with wavelength of `2 * pi`, this should
    # result in a Rayleigh range of 0.5
    w = 1.
    R = np.inf
    wavelength = 2 * np.pi
    g = GaussianBeam(None, w=w, R=R, wavelength=wavelength)

    tools.assert_almost_equal(g.q, 0 + 0.5j)
    tools.assert_almost_equal(g.R, np.inf)
    tools.assert_almost_equal(g.w, w)
    tools.assert_almost_equal(g.z, 0)
    tools.assert_almost_equal(g.zR, 0.5)

    return
