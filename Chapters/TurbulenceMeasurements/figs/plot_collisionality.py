import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct
from get_uncertainty import Uncertainty


# Shots analyzed
shots = np.array([
    171536,
    171538
])

rhos = np.array([
    0.5,
    0.8
])

# Corresponding analysis times
# [times] = ms
times = np.array([
    2750,
    2200
])

# Plotting parameters
rholim = [0.0, 1.0]
figsize = (7, 4.5)
linewidth = 2
fontsize = 16
cols = get_distinct(2)
ylim = [1, 100]
alpha = 0.5
Nsmooth = None


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


def nuei_relerr(shot, rho=None):
    Une = Uncertainty('ne', shot, rho=rho)
    UnC = Uncertainty('imp', shot, rho=rho)
    Z = 6  # Carbon charge

    Ute = Uncertainty('te', shot, rho=rho)

    # Variation in Coulomb logarithm is very weak
    # over the uncertainty scales we are considering,
    # so neglect this contribution to uncertainty
    # for simplicity...
    term1 = Une.y_relerr ** 2
    term2 = (Z * UnC.y_relerr) ** 2
    term3 = (1.5 * Ute.y_relerr) ** 2
    relerr = np.sqrt(term1 + term2 + term3)

    if rho is None:
        rho = Un.rho

    return rho, relerr


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


def vestar_relerr(shot, rho=None, a=0.56):
    Un = Uncertainty('ne', shot, rho=rho)
    Ute = Uncertainty('te', shot, rho=rho)

    # Unit conversions
    r = Un.rho * a
    Un.y *= 1e19
    Ute.y *= 1e3
    Ute.yerr *= 1e3
    Un.yperr *= (1e19 / a)
    Ute.yperr *= (1e3 / a)

    dndr = np.gradient(Un.y) / np.gradient(r)
    dtdr = np.gradient(Ute.y) / np.gradient(r)

    p = Un.y * Ute.y
    dpdr = np.gradient(p) / np.gradient(r)

    term1 = (dndr * Ute.y * Un.y_relerr) ** 2
    term2 = (Un.yperr * Ute.y) ** 2
    term3 = (dndr * Ute.yerr) ** 2
    term4 = (Un.y * Ute.yperr) ** 2

    relerr = np.sqrt(term1 + term2 + term3 + term4) / np.abs(dpdr)

    if rho is None:
        rho = Un.rho

    return rho, relerr


if __name__ == '__main__':
    a = 0.56  # [a] = m

    plt.figure(figsize=figsize)

    # Denote region inaccessible to PCI
    plt.fill_betweenx(
        ylim,
        0,
        x2=0.35,
        color='lightgray')

    for sind, shot in enumerate(shots):
        # Load spatial coordinate, rho
        fname = './%i/%ibis/rho.pkl' % (shot, times[sind])
        with open(fname, 'rb') as f:
            rho = pickle.load(f)[0, :]

        rhoind = np.where(np.logical_and(
            rho >= rholim[0],
            rho <= rholim[1]))[0]

        # Load spatial coordinate, rmin
        fname = './%i/%ibis/rmin.pkl' % (shot, times[sind])
        with open(fname, 'rb') as f:
            rmin = pickle.load(f)[0, :] * 1e-2  # [rmin] = m

        # Load profiles
        fname = './%i/%ibis/%s.pkl' % (shot, times[sind], 'te')
        with open(fname, 'rb') as f:
            Te = pickle.load(f)[0, :]  # [Te] = keV

        fname = './%i/%ibis/%s.pkl' % (shot, times[sind], 'ne')
        with open(fname, 'rb') as f:
            ne = pickle.load(f)[0, :] / 1e14  # [ne] = 10^{20} m^{-3}

        fname = './%i/%ibis/%s.pkl' % (shot, times[sind], 'ni1')
        with open(fname, 'rb') as f:
            ni = pickle.load(f)[0, :] / 1e14  # [ni] = 10^{20} m^{-3}

        nu = nuei(Te, ni) / (vestar(rmin, Te, ne) / a)

        if Nsmooth is not None:
            nu = np.convolve(
                nu,
                np.ones(Nsmooth, dtype='float') / Nsmooth,
                mode='same')

        term1 = (nuei_relerr(shot, rho=rho)[1]) ** 2
        term2 = (vestar_relerr(shot, rho=rho, a=a)[1]) ** 2
        nu_relerr = np.sqrt(term1 + term2)

        plt.semilogy(
            rho[rhoind],
            nu[rhoind],
            c=cols[sind],
            linewidth=linewidth,
            label=(r'$\mathregular{\rho_{ECH} = %.1f}$' % rhos[sind]))

        # Lower bound must remain positive
        lower_bound = np.maximum(
            (1 - nu_relerr) * nu,
            ylim[0])

        # Upper bound must remain finite...
        # Last point tends to go to infinity, so
        # let's just make it equal to 2nd-to-last data point
        upper_bound = (1 + nu_relerr) * nu
        upper_bound[-1] = upper_bound[-22]

        plt.fill_between(
            rho[rhoind],
            lower_bound[rhoind],
            upper_bound[rhoind],
            color=cols[sind],
            alpha=alpha)

    plt.xlabel(
        r'$\mathregular{\rho}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{\nu_{ei} \; [v_{*e} \,/\, a]}$',
        fontsize=fontsize)

    plt.annotate(
        r'$\mathregular{R < 1.98 \, m}$',
        (0.08, 2.5),
        fontsize=(fontsize - 3))

    # Limits and tick marks
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mpl.ticker.FormatStrFormatter('%d'))
    ax.set_yticks([1, 10, 100])
    ax.set_ylim(ylim)

    # Shot and time annotations
    for sind, shot in enumerate(shots):
        tmid = times[sind] * 1e-3
        dt = 0.1
        t0 = tmid - dt
        tf = tmid + dt

        x0 = 0.665
        y0 = 1.35
        dy = 0.25

        plt.annotate(
            '%i, [%.2f, %.2f] s' % (shot, t0, tf),
            (x0, y0 - (sind * dy)),
            color=cols[sind],
            fontsize=(fontsize - 6))

    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()
