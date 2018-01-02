import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct

from wavenumber_conversion import get_kR


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

rho = 0.60
Rpci = 1.98  # [Rpci] = m

# Plotting parameters
fontsize = 15
cols = get_distinct(2)
markers = ['o', 's']
linewidth = 2
kR_fit_lims = [4, 15]


def load_data(shot, time, rho, sat_rule=1):
    d = '%i/%ibis/SAT_RULE_%i/0%i' % (
        shot,
        time,
        sat_rule,
        np.int(np.round(100 * rho)))

    # Load data
    fname = '%s/%s.pkl' % (d, 'density_spec_1')
    with open(fname, 'rb') as f:
        density_spec = pickle.load(f)

    fname = '%s/%s.pkl' % (d, 'ky')
    with open(fname, 'rb') as f:
        ky = pickle.load(f)

    # The `density_spec` values are from
    # OMFIT['TGLF_scan']['Experimental_spectra'][0.6]['density_spectrum']['density_spec1']
    # but these values are equal to the *square root* of the values in
    # OMFIT['TGLF_scan']['Experimental_spectra'][0.6]['out.tglf.intensity_spectrum'],
    # which gives the "gyro-bohm normalized intensity fluctuation amplitude
    # spectra"... the inclusion of the word "amplitude" is a bit confusing,
    # but the use of the word "intensity" and the fact that these values
    # are the square of `density_spec` leads me to believe that
    # `density_spec` is the *amplitude* and should be squared to
    # make a power spectrum for comparison to our measurements
    Sk = density_spec ** 2

    # Package data
    data = {
        'Sk': Sk,
        'ky': ky
    }

    return data


def power_law_fit(x, y, xlim=None):
    '''Least-squares fit data to a power law, y = c * x^{alpha}.

    Parameters:
    -----------
    x - array_like, (`N`,)
        The independent variable.
        [x] = arbitrary

    ydata - array_like, (`N`,)
        The dependent variable.
        [y] = arbitrary

    xlim - array_like, (2,) or None
        If not None, only fit data within `xlim`.
        [xlim] = [x]

    Returns:
    --------
    (c, alpha) - tuple; values for power law.

    '''
    if xlim is not None:
        ind = np.where(np.logical_and(
            x >= xlim[0],
            x <= xlim[1]))[0]
    else:
        ind = slice(None, None)

    lnx = np.log(x[ind])
    lny = np.log(y[ind])

    A = np.array([lnx, np.ones(len(lnx))]).T

    fit = np.linalg.lstsq(A, lny)[0]
    alpha = fit[0]
    c = np.exp(fit[1])

    return c, alpha


if __name__ == '__main__':
    plt.figure()

    for sind, shot in enumerate(shots):
        # Load data
        d = load_data(shot, times[sind], rho, sat_rule=1)

        # Parse data
        kR = get_kR(d['ky'], rho, shot, times[sind], R0=Rpci)
        Sk = d['Sk']

        plt.loglog(
            kR,
            Sk,
            color=cols[sind],
            marker=markers[sind],
            linewidth=linewidth,
            label=r'$\mathregular{\rho_{ECH} = %.1f}$' % rho_ECH[sind])

        # Fit data from 536 to a power law
        if shot == 171536:
            c, alpha = power_law_fit(kR, Sk, xlim=kR_fit_lims)
            plt.loglog(
                kR_fit_lims,
                c * (kR_fit_lims ** alpha),
                color='k',
                linestyle='--',
                linewidth=(2 * linewidth))
            plt.annotate(
                r'$\mathregular{\propto k_R^{%.1f}}$' % alpha,
                (8.25, 0.25),
                color=cols[sind],
                fontsize=fontsize)

    # Labeling
    plt.xlabel(
        r'$\mathregular{k_R \, [cm^{-1}]}$',
        fontsize=fontsize)
    plt.ylabel(
        r'$\mathregular{S(k_R) \, [a.u.]}$',
        fontsize=fontsize)
    plt.legend(
        loc='best',
        fontsize=(fontsize - 1))

    # Limits
    xlim = [1, 200]
    ylim=[3e-3, 5]
    ylim=[1e-5, 10]
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Annotate wavenumbers outside measurable range
    fillcolor = 'gray'
    alpha = 0.5
    plt.fill_betweenx(
        ylim,
        25,
        xlim[-1],
        color=fillcolor,
        alpha=alpha)
    plt.annotate(
        r'$\mathregular{k_R > 25 \, cm^{-1}}$',
        (45, 1e-1),
        fontsize=(fontsize - 1))

    # Noise floor from PCI measurement
    plt.hlines(
        3e-2,
        xlim[0],
        xlim[1],
        linestyle='-.',
        linewidth=linewidth)
    plt.annotate(
        r'$\mathregular{inferred \, noise \, floor}$',
        (1.1, 1.5e-2),
        fontsize=fontsize)

    # Annotate radial location
    plt.annotate(
        r'$\mathregular{\rho = %.2f}$' % rho,
        (1.1, 1.3e-5),
        fontsize=(fontsize - 1))

    plt.show()
