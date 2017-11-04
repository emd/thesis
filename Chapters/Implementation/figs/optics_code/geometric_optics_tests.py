from nose import tools
import numpy as np

from ABCD_matrices import prop, lens
from geometric_optics import Ray, image_distance


def test_Ray_propagation():
    # Ray aligned with optical axis is *unaltered* by propagation
    rho0 = 0
    theta0 = 0
    r0 = Ray(rho0, theta0)

    d = 1
    r = r0.applyABCD(prop(d))
    tools.assert_almost_equal(r.rho, rho0)
    tools.assert_almost_equal(r.theta, theta0)

    # Ray displaced from but parallel to optical axis
    # is *unaltered* by propagation
    rho0 = 1
    theta0 = 0
    r0 = Ray(rho0, theta0)

    d = 1
    r = r0.applyABCD(prop(d))
    tools.assert_almost_equal(r.rho, rho0)
    tools.assert_almost_equal(r.theta, theta0)

    # Propagation does *not* change angle `theta` but does change
    # `rho` if `theta` is non-zero
    rho0 = 1
    theta0 = 1
    r0 = Ray(rho0, theta0)

    d = 1
    r = r0.applyABCD(prop(d))
    tools.assert_almost_equal(r.rho, rho0 + (d * theta0))
    tools.assert_almost_equal(r.theta, theta0)

    return


def test_Ray_focusing():
    # Ray aligned with optical axis is *unaltered* by focusing
    rho0 = 0
    theta0 = 0
    r0 = Ray(rho0, theta0)

    f = 1
    r = r0.applyABCD(lens(f))
    tools.assert_almost_equal(r.rho, rho0)
    tools.assert_almost_equal(r.theta, theta0)

    # Infinite focal length does *no* focusing
    rho0 = 1
    theta0 = 0
    r0 = Ray(rho0, theta0)

    f = np.inf
    r = r0.applyABCD(lens(f))
    tools.assert_almost_equal(r.rho, rho0)
    tools.assert_almost_equal(r.theta, theta0)

    # Finite focal length does do focusing for off-axis ray;
    # immediately after lens, the angle `theta` is altered, but
    # the transverse distance `rho` is not altered
    rho0 = 1
    theta0 = 0
    r0 = Ray(rho0, theta0)

    f = 1
    r = r0.applyABCD(lens(f))
    tools.assert_almost_equal(r.rho, rho0)
    tools.assert_almost_equal(r.theta, theta0 - (1. / f))

    return


def test_Ray_imaging():
    rho0 = 1
    theta0 = 1
    r0 = Ray(rho0, theta0)

    # One realization of a magnification M = 0.1 imaging system
    M = 0.1
    C = 0
    ABCD = np.matrix([
        [M,    0. ],
        [C, 1. / M]])

    r = r0.applyABCD(ABCD)
    tools.assert_almost_equal(r.rho, M * rho0)
    tools.assert_almost_equal(r.theta, (theta0 / M) + (C * rho0))

    return


def test_image_distance():
    # Focal length and object distances
    f = 1
    s = [2, 0.5]

    # Expected image distances
    sprime = [2, -1]

    for i in np.arange(len(s)):
        tools.assert_equal(
            image_distance(lens(f) * prop(s[i])),
            sprime[i])

    return
