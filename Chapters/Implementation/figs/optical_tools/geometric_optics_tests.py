from nose import tools
import numpy as np

from ABCD_matrices import prop, lens
from geometric_optics import image_distance


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
