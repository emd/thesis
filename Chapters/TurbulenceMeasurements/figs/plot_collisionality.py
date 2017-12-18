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
rholim = [0.0, 1.0]
figsize = (8, 8)
linewidth = 2
fontsize = 15
cols = get_distinct(2)
ylim = [1, 100]


def vte(TkeV):
    'Electron thermal velocity (m / s), from Sec. 5.1 of MFE formulary.'
    c = 3e8    # speed of light in vacuum, [c] = m / s
    me = 511.  # electron mass, [me] = keV
    return c * np.sqrt(TkeV / me)


def CoulombLog(TkeV, n20):
    'Coulomb logarithm, from Sec. 4.3.1 of MFE formulary.'
    num = 4.9e7 * (TkeV ** 1.5)
    den = np.sqrt(n20)
    return np.log(num / den)


def nuei(TkeV, n20):
    'Electron-ion collision frequency (Hz), from Sec. 4.3.1 of MFE formulary'
    num = 8.06e5 * (n20 * 1e20) * CoulombLog(TkeV, n20)
    den = (vte(TkeV)) ** 3
    return num / den


def vestar(r, TkeV, n20, B0=1.7, R0=1.7):
    'Electron diamagnetic velocity (m / s) at outboard midplane.'
    # Convert temperature to eV
    T = 1e3 * TkeV

    # Convert density to m^{-3}
    n = n20 * 1e20

    p = n * T
    dpdr = np.abs(np.gradient(p) / np.gradient(r))

    # Approximate total field with toroidal field, and
    # assume 1/R falloff from center
    B = B0 * R0 / (R0 + r)

    # Note that the electron charge that is normally
    # in the denominator *cancels* that of the "eV" in T
    den = n * B

    return dpdr / den


if __name__ == '__main__':
    a = 0.56  # [a] = m

    plt.figure()

    for sind, shot in enumerate(shots):
        # Load spatial coordinate, rho
        fname = './%i/%i/rho.pkl' % (shot, times[sind])
        with open(fname, 'rb') as f:
            rho = pickle.load(f)[0, :]

        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]

        # Load spatial coordinate, rmin
        fname = './%i/%i/rmin.pkl' % (shot, times[sind])
        with open(fname, 'rb') as f:
            rmin = pickle.load(f)[0, :] * 1e-2  # [rmin] = m

        # Load profiles
        fname = './%i/%i/%s.pkl' % (shot, times[sind], 'te')
        with open(fname, 'rb') as f:
            Te = pickle.load(f)[0, :]  # [Te] = keV

        fname = './%i/%i/%s.pkl' % (shot, times[sind], 'ne')
        with open(fname, 'rb') as f:
            ne = pickle.load(f)[0, :] / 1e14  # [ne] = 10^{20} m^{-3}

        fname = './%i/%i/%s.pkl' % (shot, times[sind], 'ni1')
        with open(fname, 'rb') as f:
            ni = pickle.load(f)[0, :] / 1e14  # [ni] = 10^{20} m^{-3}

        nu = nuei(Te, ni) / (vestar(rmin, Te, ne) / a)

        plt.semilogy(
            rho[rhoind],
            nu[rhoind],
            c=cols[sind],
            linewidth=linewidth,
            label=('%i' % shot))

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{\nu_{ei} \; [v_{*e} \,/\, a]}$',
        fontsize=fontsize)

    plt.fill_betweenx(
        ylim,
        0,
        x2=0.35,
        color='lightgray')

    plt.annotate(
        r'$\mathregular{R < 1.98 \, m}$',
        (0.08, 2.5),
        fontsize=fontsize)

    plt.legend(loc='best')
    plt.ylim(ylim)

    plt.show()
