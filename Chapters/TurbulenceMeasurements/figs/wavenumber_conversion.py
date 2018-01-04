import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct
from geometry import get_rho_hat
from geometry import load_data as load_geometry_data


shots = np.array([
    171536,
    171538
])

# [times] = ms
times = ([
    2750,
    2200,
])

rho_ECH = ([
    0.5,
    0.8
])

Rpci = 1.98  # [Rpci] = m
kRs = np.array([
    1.5,
    5.,
    25.
])

# Plotting parameters
fontsize = 16
cols = get_distinct(2)
linewidth = 2
kR_linestyles = ['-.', '--', '-']
xsec_figsize = (5, 8)
ylim = [1e-2, 1e1]


def load_data(shot, time):
    d = '%i/%ibis' % (shot, time)

    # Load data
    fname = '%s/%s.pkl' % (d, 'rho')
    with open(fname, 'rb') as f:
        rho = pickle.load(f)

    fname = '%s/%s.pkl' % (d, 'a')
    with open(fname, 'rb') as f:
        a = pickle.load(f)

    fname = '%s/%s.pkl' % (d, 'rho_s_a')
    with open(fname, 'rb') as f:
        rho_s = a * pickle.load(f)

    # Package data
    data = {
        'rho': np.squeeze(rho),
        'rho_s': np.squeeze(rho_s)
    }

    return data


def get_rho_hat_Z_slice(shot, time, R0=1.98):
    data = load_geometry_data(shot, time)
    rho_hat_R, rho_hat_Z = get_rho_hat(data)
    del rho_hat_R

    R = data['R']
    dR = np.abs(R - R0)
    Rind = np.where(dR == np.min(dR))[0][0]

    rho = data['rhoRZ'][:, Rind]
    rho_hat_Z = rho_hat_Z[:, Rind]
    Z = data['Z']

    # Only look at points inside separatrix and above midplane
    rhoind = np.where(np.logical_and(
        rho <= 1.0,
        Z >= 0))[0]

    # Only return points inside separatrix
    rho = rho[rhoind]
    rho_hat_Z = rho_hat_Z[rhoind]

    return rho, np.abs(rho_hat_Z)


def get_kR_amplification_slice(shot, time, R0=1.98):
    data = load_geometry_data(shot, time)

    # Find nearest major-radial point
    dR = np.abs(data['R'] - R0)
    Rind = np.where(dR == np.min(dR))[0][0]

    # Slice through nearest major-radial point
    rho = data['rhoRZ'][:, Rind]
    Bt = data['Bt'][:, Rind]
    Bp = data['Bp'][:, Rind]
    Z = data['Z']

    kR_amp = np.sqrt(1 + ((Bp / Bt) ** 2))

    # Only look at points inside separatrix and above midplane
    rhoind = np.where(np.logical_and(
        rho <= 1.0,
        Z >= 0))[0]

    rho = rho[rhoind]
    kR_amp = kR_amp[rhoind]

    return rho, kR_amp


def get_ky_rho_s(kR, shot, time, R0=1.98):
    'Get ky_rho_s for kR [cm^{-1}] across profile at major radius R0.'
    # Load rho_s data
    data = load_data(shot, time)
    rho = data['rho']
    rho_s = data['rho_s']

    # Compute kR amplification
    res = get_kR_amplification_slice(shot, time, R0=R0)
    rho_geo = res[0]
    kR_amp = res[1]

    # Compute rho_hat_Z (i.e. sin(theta))
    res = get_rho_hat_Z_slice(shot, time, R0=R0)
    rho_geo = res[0]
    rho_hat_Z_geo = res[1]

    # Only consider rho_s outside of minimum rho_geo
    rhoind = np.where(rho > np.min(rho_geo))[0]
    rho = rho[rhoind]
    rho_s = rho_s[rhoind]

    # Interpolate kR_amp and rho_hat_Z onto rho_s radial grid
    kR_amp = np.interp(
        rho,
        rho_geo,
        kR_amp)

    rho_hat_Z = np.interp(
        rho,
        rho_geo,
        rho_hat_Z_geo)

    ky_rho_s = kR_amp * rho_hat_Z * (kR * rho_s)

    return rho, ky_rho_s


if __name__ == '__main__':
    plt.figure()

    for sind, shot in enumerate(shots):
        time = times[sind]

        for kRind, kR in enumerate(kRs):
            rho, ky_rho_s = get_ky_rho_s(kR, shot, time)

            if kRind == 2:
                label = r'$\mathregular{\rho_{ECH} = %.1f}$' % rho_ECH[sind]
            else:
                label = None

            plt.semilogy(
                rho,
                ky_rho_s,
                color=cols[sind],
                linestyle=kR_linestyles[kRind],
                linewidth=linewidth,
                label=label)

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{|k_y \rho_s|}$',
        fontsize=fontsize)
    plt.legend(
        loc='lower left',
        fontsize=(fontsize - 1))

    plt.xlim([0, 1])
    plt.ylim(ylim)

    # Annotate each curve
    x0 = 0.575
    df = 1
    rotation = -12
    plt.annotate(
        r'$\mathregular{k_R = 1.5 \, cm^{-1}}$',
        (x0, 0.3),
        rotation=rotation,
        fontsize=(fontsize - df))
    plt.annotate(
        r'$\mathregular{k_R = 5 \, cm^{-1}}$',
        (x0, 1.),
        rotation=rotation,
        fontsize=(fontsize - df))
    plt.annotate(
        r'$\mathregular{k_R = 25 \, cm^{-1}}$',
        (x0, 4.85),
        rotation=rotation,
        fontsize=(fontsize - df))

    # Annotate region inaccessible to PCI
    plt.fill_between(
        [0, np.min(rho)],
        ylim[0],
        ylim[1],
        color='gray',
        alpha=0.5)
    plt.annotate(
        r'$\mathregular{R < 1.98 \, m}$',
        (0.065, 3e-1),
        fontsize=(fontsize - df))

    plt.show()
