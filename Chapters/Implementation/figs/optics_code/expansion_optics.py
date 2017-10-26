from ABCD_matrices import prop, lens


in2m = 0.0254  # i.e. z[m] = z[in] * in2m

# Focal lengths of focusing optics
f_exp = 10. * in2m  # [f_exp] = m
f_P1 = 80.7 * in2m  # [f_P1] = m

# Fixed distances
z_P1mid = 353.6 * in2m  # 1st parabolic mirror to midplane, [z_P1mid] = m


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
