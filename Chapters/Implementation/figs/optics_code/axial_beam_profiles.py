import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from distinct_colours import get_distinct

from ABCD_matrices import prop, lens
from geometric_optics import Ray, image_distance
from gaussian_beam import GaussianBeam, w
from expansion_optics import in2m
from imaging_optics_interferometer import (
    f_P2, f_L1, f_L2, z_midP2, z_P2L1, z_L1L2)


# Plotting parameters
cols = get_distinct(3)
fontsize = 12
dsym = r'd_{\mathcal{O}}'

# Object-plane (i.e. in-vessel waist) 1/e E radius
w0_obj = (4. / 3) * in2m  # [w0_obj] = m

# Profile resolution in axial direction
dz = 1e-2 * in2m  # [delta] = m

# Scattering wavevector
k = 5e2  # [k] = m^{-1}


# Taken from https://stackoverflow.com/a/24805549/5469497 & slightly modified
def zoomingBox(ax1, roi, ax2, color='red', linewidth=2, roiKwargs={}, arrowKwargs={}):
    '''
    **Notes (for reasons unknown to me)**
    1. Sometimes the zorder of the axes need to be adjusted manually...
    2. The figure fraction is accurate only with qt backend but not inline...
    '''
    # roiKwargs = dict([('fill',False), ('linestyle','dashed'), ('color',color), ('linewidth',linewidth)] + roiKwargs.items())
    roiKwargs = dict([('fill',False), ('linestyle','dotted'), ('color',color), ('linewidth',linewidth)] + roiKwargs.items())
    ax1.add_patch(Rectangle([roi[0],roi[2]], roi[1]-roi[0], roi[3]-roi[2], **roiKwargs))
    # arrowKwargs = dict([('arrowstyle','-'), ('color',color), ('linewidth',linewidth)] + arrowKwargs.items())
    arrowKwargs = dict([('arrowstyle','-'), ('color',color), ('linewidth',linewidth), ('linestyle','dotted')] + arrowKwargs.items())
    srcCorners = [[roi[0],roi[2]], [roi[0],roi[3]], [roi[1],roi[2]], [roi[1],roi[3]]]
    dstCorners = ax2.get_position().corners()
    srcBB = ax1.get_position()
    dstBB = ax2.get_position()
    if (dstBB.min[0]>srcBB.max[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.max[0]<srcBB.min[0] and dstBB.min[1]>srcBB.max[1]):
        src = [0, 3]; dst = [0, 3]
    elif (dstBB.max[0]<srcBB.min[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.min[0]>srcBB.max[0] and dstBB.min[1]>srcBB.max[1]):
        src = [1, 2]; dst = [1, 2]
    elif dstBB.max[1] < srcBB.min[1]:
        src = [0, 2]; dst = [1, 3]
    elif dstBB.min[1] > srcBB.max[1]:
        src = [1, 3]; dst = [0, 2]
    elif dstBB.max[0] < srcBB.min[0]:
        src = [0, 1]; dst = [2, 3]
    elif dstBB.min[0] > srcBB.max[0]:
        src = [2, 3]; dst = [0, 1]
    for k in range(2):
        ax1.annotate('', xy=dstCorners[dst[k]], xytext=srcCorners[src[k]], xycoords='figure fraction', textcoords='data', arrowprops=arrowKwargs)


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

    # First image plane, where detector is *not* placed
    z_L1image1 = image_distance(ABCD)
    d_mid_image1 = z_midP2 + z_P2L1 + z_L1image1

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
    rho_prof = np.concatenate((rho_prof, r.rho + (d * r.theta)))

    # Compute 1/e E radius and concatenate `w_prof`
    z = g.z + d
    w_prof = np.concatenate((w_prof, w(z, g.w0, g.zR)))

    # Compute cumulative ABCD
    ABCD = prop(z_L2image) * ABCD

    # Construct array of axial distance from object plane
    d = np.arange(0, len(w_prof)) * dz

    # Plot beam profiles:
    # ===================
    fig, axes = plt.subplots(2, 1)

    # Full beam profile:
    # ------------------
    # Unscattered beam
    axes[0].plot(d / in2m, w_prof / in2m, c=cols[0], label='unscattered')
    axes[0].plot(d / in2m, np.zeros(len(rho_prof)), c=cols[0])
    axes[0].plot(d / in2m, -w_prof / in2m, c=cols[0])

    # Scattered beam
    axes[0].plot(d / in2m, (rho_prof + w_prof) / in2m, c=cols[1],
                 label='scattered ($k = 5 \; \mathrm{cm}^{-1}$)')
    axes[0].plot(d / in2m, rho_prof / in2m, c=cols[1])
    axes[0].plot(d / in2m, (rho_prof - w_prof) / in2m, c=cols[1])

    # Axis parameters
    axes[0].set_ylim([-2.5, 2.5])

    # Labeling
    axes[0].set_xlabel(r'$%s \; [\mathrm{in}]$' % dsym, fontsize=fontsize)
    axes[0].set_ylabel(r'$\rho \; [\mathrm{in}]$', fontsize=fontsize)

    # Imaging-optics beam profile:
    # ----------------------------
    zoom = 80 * in2m
    dzoom_min = d[-1] - zoom
    ind = np.where(d >= dzoom_min)[0]

    # Unscattered beam
    axes[1].plot(d[ind] / in2m, w_prof[ind] / in2m, c=cols[0])
    axes[1].plot(d[ind] / in2m, np.zeros(len(rho_prof[ind])), c=cols[0])
    axes[1].plot(d[ind] / in2m, -w_prof[ind] / in2m, c=cols[0])

    # Scattered beam
    axes[1].plot(d[ind] / in2m, (rho_prof + w_prof)[ind] / in2m, c=cols[1])
    axes[1].plot(d[ind] / in2m, rho_prof[ind] / in2m, c=cols[1])
    axes[1].plot(d[ind] / in2m, (rho_prof - w_prof)[ind] / in2m, c=cols[1])

    # Labeling
    axes[1].set_xlabel(r'$%s \; [\mathrm{in}]$' % dsym, fontsize=fontsize)
    axes[1].set_ylabel(r'$\rho \; [\mathrm{in}]$', fontsize=fontsize)

    # Add focusing-optic overlays:
    # ----------------------------
    for axind, ax in enumerate(axes):
        if axind == 0:
            ax.vlines(
                z_midP2 / in2m,
                -2.5, 2.5, colors=cols[2], linewidth=2, label='focusing optic')

        ax.vlines(
            (z_midP2 + z_P2L1) / in2m,
            -1, 1, colors=cols[2], linewidth=2)
        ax.vlines(
            (z_midP2 + z_P2L1 + z_L1L2) / in2m,
            -0.75, 0.75, colors=cols[2], linewidth=2)

    # Add image-plane overlays:
    # ------------------------
    axes[0].vlines(
        d_mid_image1 / in2m, -2.5, 2.5,
        colors='k', linestyle='--', label='image plane')
    axes[0].vlines(d[-1] / in2m, -2.5, 2.5, colors='k', linestyle='--')
    axes[1].vlines(d_mid_image1 / in2m, -0.5, 0.5, colors='k', linestyle='--')
    axes[1].vlines(d[-1] / in2m, -0.5, 0.5, colors='k', linestyle='--')

    axes[1].set_xlim([315, 400])
    axes[1].set_ylim([-0.5, 0.5])

    # Add legend:
    # -----------
    axes[0].legend(ncol=1, loc='upper left', fontsize=(fontsize - 1.5))

    # Zooming:
    # --------
    # Need to call `tight_layout` *before* creating zoomed box
    plt.tight_layout()

    zoomingBox(
        axes[0],
        [315, 400, -0.5, 0.5],
        axes[1],
        color='k',
        linewidth=1.5)

    plt.show()
