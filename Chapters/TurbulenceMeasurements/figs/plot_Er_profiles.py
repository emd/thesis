import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct


# Shots analyzed
shots = np.array([
    171536,
    171538
])

# Corresponding analysis times
# [times] = ms
times = np.array([
    2750,
    2200
])

# Plotting parameters
rholim = [0.0, 0.95]
gammaE_lim = (-10, 50)
Nsmooth = 5
figsize = (9, 3.5)
linewidth = 2
fontsize = 15
cols = get_distinct(len(shots))


def get_Er_midplane(shot, time):
    # Load Er data
    d = '../doppler_shift/%i/%ibis' % (shot, time)

    fname = '%s/rho.pkl' % d
    with open(fname, 'rb') as f:
        rho = pickle.load(f)

    fname = '%s/Er_midplane.pkl' % d
    with open(fname, 'rb') as f:
        Er_midplane = pickle.load(f)

    return rho, Er_midplane


def get_gammaE(shot, time, Nsmooth=5):
    # Load Er_RBpol data
    d = '../doppler_shift/%i/%ibis' % (shot, time)

    fname = '%s/rho.pkl' % d
    with open(fname, 'rb') as f:
        rho = pickle.load(f)

    fname = '%s/Er_RBpol_midplane.pkl' % d
    with open(fname, 'rb') as f:
        Er_RBpol_midplane = pickle.load(f)

    # Load TGYRO geometry and profile data
    d = './%i/%ibis' % (shot, time)

    fname = '%s/rho.pkl' % d
    with open(fname, 'rb') as f:
        rho_TGYRO = pickle.load(f)[0, :]

    fname = '%s/rmin.pkl' % d
    with open(fname, 'rb') as f:
        rmin_TGYRO = pickle.load(f)[0, :]

    fname = '%s/q.pkl' % d
    with open(fname, 'rb') as f:
        q_TGYRO = pickle.load(f)[0, :]

    # Interpolate Er_RBpol_midplane onto TGYRO grid
    Er_RBpol_midplane_TGYRO = np.interp(
        rho_TGYRO,
        rho,
        Er_RBpol_midplane)

    # From Burrell Friday Science meeting presentation:
    #
    #   https://diii-d.gat.com/d3d-wiki/images/0/01/FSM_burrell_20170609.pdf
    #
    num = np.gradient(Er_RBpol_midplane_TGYRO)
    den = np.gradient(rmin_TGYRO)
    pre = rmin_TGYRO / q_TGYRO
    gammaE = pre * (num / den)

    if Nsmooth is not None:
        gammaE = np.convolve(
            gammaE,
            np.ones(Nsmooth, dtype='float') / Nsmooth,
            mode='same')

    return rho_TGYRO, gammaE


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=figsize)

    for shotind, shot in enumerate(shots):
        time = times[shotind]

        # Er_midplane
        rho, Er_midplane = get_Er_midplane(shot, time)
        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]
        ax[0].plot(
            rho[rhoind],
            Er_midplane[rhoind],
            color=cols[shotind],
            linewidth=linewidth)

        # gammaE
        rho, gammaE = get_gammaE(shot, time, Nsmooth=Nsmooth)
        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]
        ax[1].plot(
            rho[rhoind],
            gammaE[rhoind] / 1e3,
            color=cols[shotind],
            linewidth=linewidth,
            label=('%i, %i ms' % (shot, time)))

    ax[0].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    ax[0].set_ylabel(
        r'$\mathregular{E_r \; [kV / m]}$',
        fontsize=fontsize)
    ax[1].set_xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    ax[1].set_ylabel(
        r'$\mathregular{\gamma_E \; [kHz]}$',
        fontsize=fontsize)
    ax[1].legend(loc='best', fontsize=fontsize)

    ax[1].set_ylim(gammaE_lim)

    plt.tight_layout()
    plt.show()
