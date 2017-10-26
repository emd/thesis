from ABCD_matrices import prop, lens
from geometric_optics import image_distance
from expansion_optics import in2m, f_P1


# Focal lengths of focusing optics
f_P2 = f_P1         # [f_P2] = m
f_L1 = 7.5 * in2m   # [f_L1] = m
f_L2 = 7.5 * in2m   # [f_L2] = m

# Fixed distances
z_midP2 = 263.8 * in2m  # midplane to 2nd parabolic mirror, [z_midP2] = m
z_ref = 59.375 * in2m   # total reference-arm distance, [z_ref] = m

# Distances selected from sensitivity study
z_P2L1 = 94.6 * in2m       # [z_P2L1] = m
z_L1L2 = 23.875 * in2m     # [z_L1L2] = m


def midplane_to_detector_ABCD():
    'Get ABCD ray matrix from tokamak midplane to interferometer detector.'
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
    z_L2det = image_distance(ABCD)

    # Determine ABCD matrix for full imaging system
    ABCD = prop(z_L2det) * ABCD

    return ABCD
