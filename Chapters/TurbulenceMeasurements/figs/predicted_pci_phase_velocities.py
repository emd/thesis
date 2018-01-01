import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cPickle as pickle

from linear_stability_overview import load_data as load_stability_data
from geometry import load_data as load_geometry_data
from geometry import get_B0, get_rho_hat


shots = [171536, 171538]
times = [2750, 2200]
vpci_meas = [5.6, 6.5]  # [vpci_meas] = km / s
rhos = np.arange(0.35, 0.95, 0.05)

# Plotting parameters
figsize = (8, 4)
fontsize = 15
gamma_lim = [1, 2e3]   # [gamma_lim] = kHz
ky_lim = [0.1, 2.5]    # generous limits on PCI-measured k of fluctuation
dv_lim = [-4, 4]       # [dv_lim] = km / s


# Helper class for creating a diverging colormap with asymmetric limits;
# taken from Joe Kington at:
#
#   https://stackoverflow.com/a/20146989/5469497
#
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_plasma_frame_phase_velocities(ky_lim=ky_lim):
    fig, axs = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=figsize)

    for sind, shot in enumerate(shots):
        # Load data
        time = times[sind]
        data = load_stability_data(shot, time, rhos, physical_units=True)

        # Parse data
        # (ky * rho_s varies only very slightly in rho, so
        # just take the values at rhos[0])
        ky = data['ky'][:, 0]  # [ky] = 1 / rho_s
        vph = data['vph']      # [vph] = km / s

        # Only examine data within PCI krange
        kind = np.where(np.logical_and(
            ky >= ky_lim[0],
            ky <= ky_lim[1]))[0]

        m = axs[sind].pcolormesh(
            rhos,
            ky[kind],
            vph[kind, :],
            cmap='BrBG',
            norm=MidpointNormalize(midpoint=0))
        cb = plt.colorbar(m, ax=axs[sind])
        m.set_edgecolor('face')  # avoid grid lines in PDF file
        cb.set_label(
            r'$\mathregular{v_{ph} \; [km / s]}$',
            fontsize=fontsize)

    # Plot limits and scale
    axs[0].set_xlim([rhos[0], rhos[-1]])
    axs[0].set_ylim([ky[kind][0], ky[kind][-1]])

    # Labeling
    axs[0].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[1].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[0].set_ylabel(
        r'$\mathregular{k_y \rho_s}$',
        fontsize=fontsize)
    axs[0].set_title(
        r'$\mathregular{\rho_{ECH} = 0.5}$',
        fontsize=fontsize)
    axs[1].set_title(
        r'$\mathregular{\rho_{ECH} = 0.8}$',
        fontsize=fontsize)

    plt.tight_layout()
    plt.show()

    return


def get_kR_amplification(shot, time, rhointerp=rhos, R0=1.98):
    # Load data
    data = load_geometry_data(shot, time)

    # Find nearest major-radial point
    dR = np.abs(data['R'] - R0)
    Rind = np.where(dR == np.min(dR))[0][0]

    # Slice through nearest major-radial point
    rho = data['rhoRZ'][:, Rind]
    Bt = data['Bt'][:, Rind]
    B0 = get_B0(data)[:, Rind]
    Z = data['Z']

    kR_amp = B0 / Bt

    # Only look at points inside separatrix and above midplane
    rhoind = np.where(np.logical_and(
        rho <= 1.0,
        Z >= 0))[0]

    rho = rho[rhoind]
    kR_amp = kR_amp[rhoind]

    if rhointerp is not None:
        kR_amp = np.interp(rhointerp, rho, kR_amp)
    else:
        rhointerp = rho

    return rhointerp, kR_amp


def plot_kR_amplification(rhointerp=rhos):
    plt.figure()

    for sind, shot in enumerate(shots):
        rho, kR_amp = get_kR_amplification(
            shot,
            times[sind],
            rhointerp=rhointerp)

        plt.plot(rho, kR_amp, label=('%i' % shot))

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{B_0 / B_{\zeta,0}}$',
        fontsize=fontsize)
    plt.legend(loc='best')

    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)

    plt.tight_layout()
    plt.show()

    return


def get_R_projection(shot, time, rhointerp=rhos, R0=1.98):
    # Load data
    data = load_geometry_data(shot, time)

    # Get rho_hat_Z (= sin(theta))
    rho_hat_R, rho_hat_Z = get_rho_hat(data)
    del rho_hat_R

    # Find nearest major-radial point
    dR = np.abs(data['R'] - R0)
    Rind = np.where(dR == np.min(dR))[0][0]

    # Slice through nearest major-radial point
    rho = data['rhoRZ'][:, Rind]
    rho_hat_Z = rho_hat_Z[:, Rind]
    Z = data['Z']

    # Only look at points inside separatrix and above midplane
    rhoind = np.where(np.logical_and(
        rho <= 1.0,
        Z >= 0))[0]

    rho = rho[rhoind]
    rho_hat_Z = rho_hat_Z[rhoind]

    if rhointerp is not None:
        rho_hat_Z = np.interp(rhointerp, rho, rho_hat_Z)
    else:
        rhointerp = rho

    return rhointerp, rho_hat_Z


def plot_R_projection(rhointerp=rhos):
    plt.figure()

    for sind, shot in enumerate(shots):
        rho, Rproj = get_R_projection(
            shot,
            times[sind],
            rhointerp=rhointerp)

        plt.plot(rho, Rproj, label=('%i' % shot))

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{\hat{\rho} \cdot \hat{z} = sin\theta}$',
        fontsize=fontsize)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    return


def load_Er_data(shot, time):
    d = '%i/%ibis/Er' % (shot, time)

    # Load data
    fname = '%s/%s.pkl' % (d, 'rho')
    with open(fname, 'rb') as f:
        rho = pickle.load(f)

    fname = '%s/%s.pkl' % (d, 'Er_RBpol_midplane')
    with open(fname, 'rb') as f:
        Er_RBpol_midplane = pickle.load(f)

    # Package data
    data = {
        'rho': rho,
        'Er_RBpol_midplane': Er_RBpol_midplane
    }

    return data


def get_vExB(shot, time, rhointerp=rhos, R0=1.98):
    Er_data = load_Er_data(shot, time)
    geometry_data = load_geometry_data(shot, time)

    # Get magnetic-field magnitude
    B0 = get_B0(geometry_data)

    # Er_RBpol is a *flux* function. Interpolate outboard
    # midplane values onto rho(R,Z) grid from G-file, and
    # assign value 0 to points with rho > 1.
    Er_RBpol_RZ = np.interp(
        geometry_data['rhoRZ'],
        Er_data['rho'],
        Er_data['Er_RBpol_midplane'],
        right=0)

    # Note that the resulting `Er` has units of kV / m; which
    # can be seen be verified by running the GUI in
    #
    #   OMFIT['TGLF_scan']['GUIS']['tglf_scan_gui']
    #
    # going to the 'Setup input profiles' tab, and then
    # clicking the 'Plot radial electric field' button;
    # compare the plotted profile (units of kV / m) to
    # that obtained in `Er` computed below.
    R = geometry_data['R'][np.newaxis, :]  # need extra dimension...
    Bp = geometry_data['Bp']
    Er = (R * Bp) * Er_RBpol_RZ
    Er /= 1e3  # convert to kV / m

    # Find nearest major-radial point
    R = np.squeeze(R)  # remove extra dimension
    dR = np.abs(R - R0)
    Rind = np.where(dR == np.min(dR))[0][0]

    # Slice through nearest major-radial point
    rho = geometry_data['rhoRZ'][:, Rind]
    Er = Er[:, Rind]
    B0 = B0[:, Rind]
    Z = geometry_data['Z']

    # Only look at points inside separatrix and above midplane
    rhoind = np.where(np.logical_and(
        rho <= 1.0,
        Z >= 0))[0]

    rho = rho[rhoind]
    Er = Er[rhoind]
    B0 = B0[rhoind]

    vExB = Er / B0

    if rhointerp is not None:
        vExB = np.interp(rhointerp, rho, vExB)
    else:
        rhointerp = rho

    return rhointerp, vExB


def plot_vExB(rhointerp=rhos):
    plt.figure()

    for sind, shot in enumerate(shots):
        rho, vExB = get_vExB(
            shot,
            times[sind],
            rhointerp=rhointerp)

        plt.plot(rho, vExB, label=('%i' % shot))

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{v_E \, [km / s]}$',
        fontsize=fontsize)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    # # Testing of individual routines
    # plot_plasma_frame_phase_velocities(ky_lim=ky_lim)
    # rhointerp = rhos
    # plot_kR_amplification(rhointerp=rhointerp)
    # plot_R_projection(rhointerp=rhointerp)
    # plot_vExB(rhointerp=rhointerp)

    fig, axs = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=figsize)

    for sind, shot in enumerate(shots):
        # Load linear-stability data
        time = times[sind]
        data = load_stability_data(shot, time, rhos, physical_units=True)

        # Parse data
        # (ky * rho_s varies only very slightly in rho, so
        # just take the values at rhos[0])
        ky = data['ky'][:, 0]  # [ky] = 1 / rho_s
        vph = data['vph']      # [vph] = km / s

        # Iteratively build up predicted PCI phase velocity
        vpci = -vph
        vpci += get_vExB(shot, time, rhointerp=rhos)[1]
        vpci *= get_kR_amplification(shot, time, rhointerp=rhos)[1]
        vpci *= get_R_projection(shot, time, rhointerp=rhos)[1]

        # Only examine magnitude
        vpci = np.abs(vpci)

        # ... and compare to measured phase velocity
        dv = vpci - vpci_meas[sind]

        # Only examine data within PCI krange
        kind = np.where(np.logical_and(
            ky >= ky_lim[0],
            ky <= ky_lim[1]))[0]

        m = axs[sind].pcolormesh(
            rhos,
            ky[kind],
            dv[kind, :],
            vmin=dv_lim[0],
            vmax=dv_lim[1],
            cmap='BrBG')
        cb = plt.colorbar(m, ax=axs[sind])
        m.set_edgecolor('face')  # avoid grid lines in PDF file
        cb.set_label(
            r'$\mathregular{\delta v \; [km / s]}$',
            fontsize=fontsize)
        cb.set_ticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    # Plot limits and scale
    axs[0].set_xlim([rhos[0], rhos[-1]])
    axs[0].set_ylim([ky[kind][0], ky[kind][-1]])

    # Labeling
    axs[0].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[1].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    axs[0].set_ylabel(
        r'$\mathregular{k_y \rho_s}$',
        fontsize=fontsize)
    axs[0].set_title(
        r'$\mathregular{\rho_{ECH} = 0.5}$',
        fontsize=fontsize)
    axs[1].set_title(
        r'$\mathregular{\rho_{ECH} = 0.8}$',
        fontsize=fontsize)

    plt.tight_layout()
    plt.show()
