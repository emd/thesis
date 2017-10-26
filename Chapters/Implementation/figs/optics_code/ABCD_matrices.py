import numpy as np


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
