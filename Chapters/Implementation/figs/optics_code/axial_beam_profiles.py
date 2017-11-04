import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct

from ABCD_matrices import prop, lens
from geometric_optics import Ray, image_distance
from gaussian_beam import GaussianBeam, w
from expansion_optics import in2m
from imaging_optics_interferometer import (
    f_P2, f_L1, f_L2, z_midP2, z_P2L1, z_L1L2)


# Plotting parameters
cols = get_distinct(2)

# Object-plane (i.e. in-vessel waist) 1/e E radius
w0_obj = (4. / 3) * in2m  # [w0_obj] = m

# Profile resolution in axial direction
dz = 1e-2 * in2m  # [delta] = m

# Scattering wavevector
k = 5e2  # [k] = m^{-1}


if __name__ == '__main__':
    # Object plane:
    # -------------
    # Object-plane Gaussian beam
    g_obj = GaussianBeam(None, w=w0_obj, R=np.inf)

    # Object-plane ray corresponding to symmetry axis of scattered beam
    k0 = 2 * np.pi / g_obj.wavelength
    r_obj = Ray(0, k / k0)

    # Initialize empty arrays for profiles
    w_prof = np.array([])
    rho_prof = np.array([])

    # Initialize ABCD ray matrix, which will be cumulative ABCD
    # from object plane to the latest optical element
    ABCD = np.identity(2)

    g = g_obj.applyABCD(ABCD)
    r = r_obj.applyABCD(ABCD)

    # Propagate from object plane to P2:
    # ----------------------------------
    # Compute ray trajectory and concatenate `rho_prof`
    d = np.arange(0, z_midP2 + dz, dz)
    rho_prof = np.concatenate((rho_prof, r.rho + (d * r.theta)))

    # Compute 1/e E radius and concatenate `w_prof`
    z = g.z + d
    w_prof = np.concatenate((w_prof, w(z, g.w0, g.zR)))

    # Compute cumulative ABCD
    ABCD = prop(z_midP2) * ABCD

    # Focus with P2:
    # --------------
    ABCD = lens(f_P2) * ABCD
    g = g_obj.applyABCD(ABCD)
    r = r_obj.applyABCD(ABCD)

    # Propagate from P2 to L1:
    # ------------------------
    # Compute ray trajectory and concatenate `rho_prof`
    d = np.arange(0, z_P2L1 + dz, dz)
    rho_prof = np.concatenate((rho_prof, r.rho + (d * r.theta)))

    # Compute 1/e E radius and concatenate `w_prof`
    z = g.z + d
    w_prof = np.concatenate((w_prof, w(z, g.w0, g.zR)))

    # Compute cumulative ABCD
    ABCD = prop(z_P2L1) * ABCD

    # Focus with L1:
    # --------------
    ABCD = lens(f_L1) * ABCD
    g = g_obj.applyABCD(ABCD)
    r = r_obj.applyABCD(ABCD)

    # Propagate from L1 to L2:
    # ------------------------
    # Compute ray trajectory and concatenate `rho_prof`
    d = np.arange(0, z_L1L2 + dz, dz)
    rho_prof = np.concatenate((rho_prof, r.rho + (d * r.theta)))

    # Compute 1/e E radius and concatenate `w_prof`
    z = g.z + d
    w_prof = np.concatenate((w_prof, w(z, g.w0, g.zR)))

    # Compute cumulative ABCD
    ABCD = prop(z_L1L2) * ABCD

    # Focus with L2:
    # --------------
    ABCD = lens(f_L2) * ABCD
    g = g_obj.applyABCD(ABCD)
    r = r_obj.applyABCD(ABCD)

    # Propagate from L2 to image plane:
    # ---------------------------------
    z_L2image = image_distance(ABCD)

    # Compute ray trajectory and concatenate `rho_prof`
    d = np.arange(0, z_L2image + dz, dz)
    # d = np.arange(0, z_L2image, dz)
    rho_prof = np.concatenate((rho_prof, r.rho + (d * r.theta)))

    # Compute 1/e E radius and concatenate `w_prof`
    z = g.z + d
    w_prof = np.concatenate((w_prof, w(z, g.w0, g.zR)))

    # Compute cumulative ABCD
    ABCD = prop(z_L2image) * ABCD

    # Construct array of axial distance from image plane
    d = np.arange(0, len(w_prof)) * dz
    d -= d[-1]

    # Plot beam profiles:
    # ===================
    fig, axes = plt.subplots(2, 1)

    # Full beam profile:
    # ------------------
    # Unscattered beam
    axes[0].plot(d / in2m, w_prof / in2m, c=cols[0])
    axes[0].plot(d / in2m, np.zeros(len(rho_prof)), c=cols[0])
    axes[0].plot(d / in2m, -w_prof / in2m, c=cols[0])

    # Scattered beam
    axes[0].plot(d / in2m, (rho_prof + w_prof) / in2m, c=cols[1])
    axes[0].plot(d / in2m, rho_prof / in2m, c=cols[1])
    axes[0].plot(d / in2m, (rho_prof - w_prof) / in2m, c=cols[1])

    # Axis parameters
    axes[0].set_ylim([-2.5, 2.5])

    # Imaging-optics beam profile:
    # ----------------------------
    d_zoom = 80 * in2m
    ind = np.where((d + d_zoom) >= 0)[0]

    # Unscattered beam
    axes[1].plot(d[ind] / in2m, w_prof[ind] / in2m, c=cols[0])
    axes[1].plot(d[ind] / in2m, np.zeros(len(rho_prof[ind])), c=cols[0])
    axes[1].plot(d[ind] / in2m, -w_prof[ind] / in2m, c=cols[0])

    # Scattered beam
    axes[1].plot(d[ind] / in2m, (rho_prof + w_prof)[ind] / in2m, c=cols[1])
    axes[1].plot(d[ind] / in2m, rho_prof[ind] / in2m, c=cols[1])
    axes[1].plot(d[ind] / in2m, (rho_prof - w_prof)[ind] / in2m, c=cols[1])

    plt.show()
