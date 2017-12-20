import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cPickle as pickle
from distinct_colours import get_distinct

from geometry import load_data as load_geometry_data
from geometry import (
    get_B0,
    get_rho_hat,
    xsec_figsize)


# [Rpci] = m
Rpci = 1.98

# PCI-measured phase velocities
vmeas_536 = 5.65  # [vmeas_536] = km / s
vmeas_538 = 6.47  # [vmeas_538] = km / s

# Plotting parameters
figsize = (7, 5)
fontsize = 15
linewidth = 2
cols = get_distinct(2)
linestyle_vpci = '-'
linestyle_vmeas = '--'
rholim = [0.3, 1.0]
rholim_window = [0.3, 0.95]
alpha = 0.5


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


def load_Er_data(shot, time):
    d = '%i/%ibis/' % (shot, time)

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


def get_Er(shot, time):
    Er_data = load_Er_data(shot, time)
    geometry_data = load_geometry_data(shot, time)

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
    R = geometry_data['R'][np.newaxis, :]
    Bp = geometry_data['Bp']
    Er = (R * Bp) * Er_RBpol_RZ
    Er /= 1e3  # convert to kV / m

    return Er


def plot_Er(shot, time, cmap='RdBu', R=Rpci, N=30):
    geometry_data = load_geometry_data(shot, time)
    Er = get_Er(shot, time)

    plt.figure(figsize=xsec_figsize)
    plt.plot(
        geometry_data['Rlim'],
        geometry_data['Zlim'],
        c='k',
        linewidth=linewidth)
    plt.gca().set_aspect('equal')
    plt.contour(
        geometry_data['R'],
        geometry_data['Z'],
        geometry_data['rhoRZ'],
        np.arange(0, 1.1, 0.1),
        colors='k',
        linewidths=0.5)
    plt.contourf(
        geometry_data['R'],
        geometry_data['Z'],
        Er,
        N,
        cmap=cmap,
        norm=MidpointNormalize(midpoint=0))
    plt.colorbar()
    plt.axvline(R, c='r', linewidth=linewidth)
    plt.title(
        r'$\mathregular{E_r \; [kV/m]}$',
        fontsize=fontsize)
    plt.show()

    return


def get_vExB(shot, time):
    geometry_data = load_geometry_data(shot, time)
    B0 = get_B0(geometry_data)

    Er = get_Er(shot, time)

    # Compute ExB velocity
    # [vExB] = km / s
    vExB = Er / B0

    return vExB


def get_vpci(shot, time):
    geometry_data = load_geometry_data(shot, time)
    Bt = geometry_data['Bt']
    Bp = geometry_data['Bp']
    B0 = get_B0(geometry_data)

    Er = get_Er(shot, time)

    # Compute z-component, rho_hat_Z, of flux-surface normal
    # along PCI beam path
    rho_hat_Z = get_rho_hat(geometry_data)[1]

    # Compute PCI-measured velocity, noting that
    # rho_hat_z = sin(theta), where theta is the angle
    # between rho_hat and Rhat (i.e. the unit vectors
    # in the rho direction and the major-radial direction,
    # respectively)
    # [vpci] = km / s
    num = Er * rho_hat_Z
    den = np.sqrt((Bt ** 2) + ((Bp * rho_hat_Z) ** 2))
    vpci = num / den

    return vpci


def plot_vExB(shot, time, cmap='RdBu', R=Rpci, N=30):
    geometry_data = load_geometry_data(shot, time)
    vExB = get_vExB(shot, time)

    plt.figure(figsize=xsec_figsize)
    plt.plot(
        geometry_data['Rlim'],
        geometry_data['Zlim'],
        c='k',
        linewidth=linewidth)
    plt.gca().set_aspect('equal')
    plt.contour(
        geometry_data['R'],
        geometry_data['Z'],
        geometry_data['rhoRZ'],
        np.arange(0, 1.1, 0.1),
        colors='k',
        linewidths=0.5)
    plt.contourf(
        geometry_data['R'],
        geometry_data['Z'],
        vExB,
        N,
        norm=MidpointNormalize(midpoint=0),
        cmap=cmap)
    plt.colorbar()
    plt.axvline(R, c='r', linewidth=linewidth)
    plt.title(
        r'$\mathregular{v_E \; [km/s]}$',
        fontsize=fontsize)
    plt.show()

    return


def plot_vpci(shot, time, cmap='RdBu', R=Rpci, N=30):
    geometry_data = load_geometry_data(shot, time)
    vpci = get_vpci(shot, time)

    plt.figure(figsize=xsec_figsize)
    plt.plot(
        geometry_data['Rlim'],
        geometry_data['Zlim'],
        c='k',
        linewidth=linewidth)
    plt.gca().set_aspect('equal')
    plt.contour(
        geometry_data['R'],
        geometry_data['Z'],
        geometry_data['rhoRZ'],
        np.arange(0, 1.1, 0.1),
        colors='k',
        linewidths=0.5)
    plt.contourf(
        geometry_data['R'],
        geometry_data['Z'],
        vpci,
        N,
        norm=MidpointNormalize(midpoint=0),
        cmap=cmap)
    plt.colorbar()
    plt.axvline(R, c='r', linewidth=linewidth)
    plt.title(
        r'$\mathregular{v_{pci} \; [km/s]}$',
        fontsize=fontsize)
    plt.show()

    return


def get_velocity_profiles(shot, time, R=Rpci):
    geometry_data = load_geometry_data(shot, time)

    # Take a slice through nearest major-radial point
    dR = np.abs(geometry_data['R'] - R)
    Rind = np.where(dR == np.min(dR))[0][0]

    rho = geometry_data['rhoRZ'][:, Rind]
    vpci = get_vpci(shot, time)[:, Rind]
    vExB = get_vExB(shot, time)[:, Rind]

    # Only return points inside separatrix
    rhoind = np.where(rho <= 1.0)[0]

    return rho[rhoind], vpci[rhoind], vExB[rhoind]


if __name__ == '__main__':
    # Compute velocity profiles for 171536
    shot = 171536
    time = 2750
    res_536 = get_velocity_profiles(
        shot, time, R=Rpci)
    rho_536 = res_536[0]
    vpci_536 = res_536[1]

    # Account for Er uncertainty
    dname = '../profiles/171536/2750bis/uncertainties/gaprofiles'
    fname = '%s/Er_relerr.pkl' % dname

    with open(fname, 'rb') as f:
        rho_Er = pickle.load(f)
        Er_relerr = pickle.load(f)

    Er_relerr = np.interp(rho_536, rho_Er, Er_relerr)

    # Plot:
    # -----
    plt.figure(figsize=figsize)

    rhoind = np.where(np.logical_and(
        rho_536 >= rholim[0],
        rho_536 <= rholim[1]))[0]

    # 536
    col_536 = cols[0]
    plt.plot(
        rho_536[rhoind],
        vpci_536[rhoind],
        c=col_536,
        linewidth=linewidth,
        linestyle=linestyle_vpci,
        label=r'$\mathregular{v_{pci}^E}$')
    plt.fill_between(
        rho_536[rhoind],
        ((1 - Er_relerr) * vpci_536)[rhoind],
        ((1 + Er_relerr) * vpci_536)[rhoind],
        color=col_536,
        alpha=alpha)
    plt.axhline(
        vmeas_536,
        c=col_536,
        linewidth=linewidth,
        linestyle=linestyle_vmeas,
        label=r'$\mathregular{v_{pci}^{meas}}$')
    plt.axhline(
        -vmeas_536,
        c=col_536,
        linewidth=linewidth,
        linestyle=linestyle_vmeas)

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(r'$\mathregular{v \; [km / s]}$',
        fontsize=fontsize)
    plt.legend(loc='upper left', ncol=2, fontsize=fontsize)

    tmid = time * 1e-3
    dt = 0.1
    plt.annotate(
        '%i, [%.2f, %.2f] s' % (shot, tmid - dt, tmid + dt),
        (0.31, -9.5),
        color=cols[0],
        fontsize=(fontsize - 4))

    plt.xlim(rholim_window)
    plt.ylim([-10, 10])

    plt.show()
