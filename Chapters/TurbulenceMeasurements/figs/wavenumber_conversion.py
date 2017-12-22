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

# Plotting parameters
fontsize = 16
cols = get_distinct(2)
linewidth = 2
xsec_figsize = (5, 8)


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
    Rind = np.where(dR == np.min(dR))[0]

    rho = data['rhoRZ'][:, Rind]
    rho_hat_Z = rho_hat_Z[:, Rind]

    # Only return points inside separatrix
    rhoind = np.where(rho <= 1.0)[0]
    rho = rho[rhoind]
    rho_hat_Z = rho_hat_Z[rhoind]

    # Also, we want a monotonic rho
    # (near symmetric above & below midplane)
    rhoind = np.where(rho == np.min(rho))[0][0]
    rho = np.squeeze(rho[rhoind:])  # rho monotonically increases here
    rho_hat_Z = np.squeeze(rho_hat_Z[rhoind:])

    return rho, np.abs(rho_hat_Z)


def get_ky_rho_s(kR, shot, time, R0=1.98):
    'Get ky_rho_s for kR [cm^{-1}] across profile at major radius R0.'
    data = load_data(shot, time)
    rho = data['rho']
    rho_s = data['rho_s']

    res = get_rho_hat_Z_slice(shot, time, R0=R0)
    rho_geo = res[0]
    rho_hat_Z_geo = res[1]

    rhoind = np.where(rho > np.min(rho_geo))[0]
    rho = rho[rhoind]
    rho_s = rho_s[rhoind]

    rho_hat_Z = np.interp(
        rho,
        rho_geo,
        rho_hat_Z_geo)

    ky_rho_s = kR * rho_hat_Z * rho_s

    return rho, ky_rho_s


if __name__ == '__main__':
    plt.figure()

    for sind, shot in enumerate(shots):
        time = times[sind]

        # PCI
        rho, ky_rho_s = get_ky_rho_s(1.5, shot, time)
        plt.plot(
            rho,
            ky_rho_s,
            color=cols[sind],
            linewidth=linewidth,
            label=r'$\mathregular{\rho_{ECH} = %.1f}$' % rho_ECH[sind])

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{k_y \rho_s}$',
        fontsize=fontsize)
    plt.title(
        r'$\mathregular{k_R = 1.5 \, cm^{-1}}$',
        fontsize=(fontsize - 1))
    plt.legend(
        loc='upper right',
        fontsize=(fontsize - 1))

    plt.show()
