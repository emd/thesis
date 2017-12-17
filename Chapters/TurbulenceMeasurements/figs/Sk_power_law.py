import numpy as np
import matplotlib as mpl
# mpl.use('Agg')  # no X11 on worker nodes, so use non-interactive backend
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct

import random_data as rd
import mitpci
import filters


# Spectral-estimation parameters
p = 6
Nk = 1000

# Plotting parameters
cols = get_distinct(2)
cmap = 'viridis'
flim = [10, 1500]  # [flim] = kHz
vlim_Skf = [3e-10, 1e-4]
fontsize = 14
linewidth = 2


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
    # Load previously computed correlation functions
    # ==============================================
    fname = './corrs.pkl'
    with open(fname, 'rb') as f:
        corr_list = pickle.load(f)
        rho_list = pickle.load(f)
        shot_list = pickle.load(f)
        tlim_list = pickle.load(f)
        tau_list = pickle.load(f)
        Nreal_list = pickle.load(f)

    shot_list = np.array(shot_list)

    ind_536 = np.where(shot_list == 171536)[0][0]
    ind_538 = np.where(shot_list == 171538)[0][0]

    # Compute 2d autospectral densities and plot
    # ==========================================
    asd2d_536 = mitpci.pci.TwoDimensionalAutoSpectralDensity(
        corr_list[ind_536], burg_params={'p': p, 'Nk': Nk})
    asd2d_538 = mitpci.pci.TwoDimensionalAutoSpectralDensity(
        corr_list[ind_538], burg_params={'p': p, 'Nk': Nk})

    descriptors_536 = (
        171536,
        np.int(rho_list[ind_536] * 10),
        np.int(tlim_list[ind_536][0] * 1e3),
        np.int(tlim_list[ind_536][1] * 1e3),
        Nreal_list[ind_536],
        p)
    descriptors_538 = (
        171538,
        np.int(rho_list[ind_538] * 10),
        np.int(tlim_list[ind_538][0] * 1e3),
        np.int(tlim_list[ind_538][1] * 1e3),
        Nreal_list[ind_538],
        p)

    fname_536 = ('./Skf_%i_rho0%i_%i_to_%i_Nreal%i_p%i.png' % descriptors_536)
    fname_538 = ('./Skf_%i_rho0%i_%i_to_%i_Nreal%i_p%i.png' % descriptors_538)

    asd2d_536.plotSpectralDensity(flim=flim, vlim=vlim_Skf)
    plt.text(-25, 1440, '171536', color='white')
    plt.savefig(fname_536)
    plt.close()

    asd2d_538.plotSpectralDensity(flim=flim, vlim=vlim_Skf)
    plt.text(-25, 1440, '171538', color='white')
    plt.savefig(fname_538)
    plt.close()

    # Look at wavenumber structure of individual branches:
    # ====================================================

    # Determine "source" and "destination" points of line profiles:
    # -------------------------------------------------------------
    # Points in each branch
    pt_536 = (10., 900.)
    pt_538 = (6.8, 700.)

    # Assume that each branch passes through origin
    src_536 = (0, 0)
    src_538 = (0, 0)

    # Slopes...
    m_536 = pt_536[1] / pt_536[0]
    m_538 = pt_538[1] / pt_538[0]

    # Destination points
    dst_536 = (flim[1] / m_536, flim[1])
    dst_538 = (flim[1] / m_538, flim[1])

    # Overplot branch boundaries on S(k,f):
    # -------------------------------------
    # S(k, f) => k in row space, f in column space
    #
    # NOTE: Specifying `lwr` keyword means that `coord_lines`
    # will be uniformly spaced in k and that averaging will
    # be done over a narrow range in k at each frequency.
    # To get S(k) we technically want the reverse -- averaging
    # over a narrow range in f at each wavenumber. However,
    # because the branch "bends", particularly near the low-k
    # cutoff of the PCI, this technically correct method
    # introduces artifacts into the S(k)... Because we're
    # examining a narrow slice in S(k,f), it is OK to use
    # the former method of averaging over narrow range in k
    # at each frequency, especially as it allows us to avoid
    # the "bend" biasing.
    lpc_kwargs = {'N': 100, 'lwr': 2.5, 'L': 11}

    coord_lines_536 = rd.utilities.line_profile_coordinates(
        src_536, dst_536, **lpc_kwargs)
    coord_lines_538 = rd.utilities.line_profile_coordinates(
        src_538, dst_538, **lpc_kwargs)

    fname_536 = ('./Skf_%i_rho0%i_%i_to_%i_Nreal%i_p%i_branch.png' % descriptors_536)
    fname_538 = ('./Skf_%i_rho0%i_%i_to_%i_Nreal%i_p%i_branch.png' % descriptors_538)

    asd2d_536.plotSpectralDensity(flim=flim, vlim=vlim_Skf)
    plt.text(-25, 1440, '171536', color='white')
    plt.plot(
        coord_lines_536[0, :, 0],
        coord_lines_536[1, :, 0],
        c='fuchsia')
    plt.plot(
        coord_lines_536[0, :, -1],
        coord_lines_536[1, :, -1],
        c='fuchsia')
    plt.xlim(asd2d_536.k[0] / 1e2, asd2d_536.k[-1] / 1e2)
    plt.ylim(flim)
    plt.savefig(fname_536)

    asd2d_538.plotSpectralDensity(flim=flim, vlim=vlim_Skf)
    plt.text(-25, 1440, '171538', color='white')
    plt.plot(
        coord_lines_538[0, :, 0],
        coord_lines_538[1, :, 0],
        c='fuchsia')
    plt.plot(
        coord_lines_538[0, :, -1],
        coord_lines_538[1, :, -1],
        c='fuchsia')
    plt.xlim(asd2d_538.k[0] / 1e2, asd2d_538.k[-1] / 1e2)
    plt.ylim(flim)
    plt.savefig(fname_538)

    # Compute wavenumber profiles:
    # ----------------------------
    res = rd.utilities.line_profile(
        asd2d_536.Sxx, asd2d_536.k / 1e2, asd2d_536.f / 1e3,
        src_536, dst_536,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 1})
    Sxx_prof_536 = res[0]
    k_prof_536 = res[1]
    f_prof_536 = res[2]

    res = rd.utilities.line_profile(
        asd2d_538.Sxx, asd2d_538.k / 1e2, asd2d_538.f / 1e3,
        src_538, dst_538,
        lpc_kwargs=lpc_kwargs, mc_kwargs={'order': 1})
    Sxx_prof_538 = res[0]
    k_prof_538 = res[1]
    f_prof_538 = res[2]

    # Convert from rad^2 / (Hz * m^{-1}) to rad^2 / (kHz * cm^{-1})
    Sxx_prof_536 *= (1e3 * 1e2)
    Sxx_prof_538 *= (1e3 * 1e2)

    # And account for "integration" over frequency by multiplying by `df`
    # such that the units become rad^2 / cm^{-1}
    df = 250.  # [df] = kHz
    Sxx_prof_536 *= df
    Sxx_prof_538 *= df

    # Plot wavenumber profiles:
    # -------------------------
    kind_536 = np.where(k_prof_536 >= 1.5)[0]
    kind_538 = np.where(k_prof_538 >= 1.5)[0]

    plt.figure()

    plt.loglog(
        k_prof_536[kind_536],
        Sxx_prof_536[kind_536],
        label=r'$\mathregular{ECH \; @ \; \rho = 0.5}$',
        linewidth=linewidth,
        c=cols[0])
    plt.loglog(
        k_prof_538[kind_538],
        Sxx_prof_538[kind_538],
        label=r'$\mathregular{ECH \; @ \; \rho = 0.8}$',
        linewidth=linewidth,
        c=cols[1])

    plt.legend(loc='best')

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_xlim([1, 20])
    ax.set_xlabel(
        r'$\mathregular{k_R \; [cm^{-1}]}$',
        fontsize=(fontsize + 2))
    ax.set_ylabel(
        r'$\mathregular{S_{\phi,\phi}(k_R) \; [rad^2 /\, cm^{-1}]}$',
        fontsize=(fontsize + 2))

    # Fit wavenumber profiles:
    # ------------------------
    klim_fit = np.array([4., 8.])

    c_536, alpha_536 = power_law_fit(
        k_prof_536, Sxx_prof_536, xlim=klim_fit)
    c_538, alpha_538 = power_law_fit(
        k_prof_538, Sxx_prof_538, xlim=klim_fit)

    cfit_536 = cols[0]
    cfit_538 = cols[1]

    plt.text(
        6.5, 4e-6,
        '$\mathregular{\propto k_R^{%.1f}}$' % alpha_536,
        fontsize=(fontsize + 2),
        color=cfit_536)
    plt.loglog(
        klim_fit,
        c_536 * (klim_fit ** alpha_536),
        linestyle='--',
        linewidth=(2 * linewidth),
        c='k')

    plt.text(
        3.5, 4e-6,
        '$\mathregular{\propto k_R^{%.1f}}$' % alpha_538,
        fontsize=(fontsize + 2),
        color=cfit_538)
    plt.loglog(
        klim_fit,
        c_538 * (klim_fit ** alpha_538),
        linestyle='--',
        linewidth=(2 * linewidth),
        c='k')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mpl.ticker.FormatStrFormatter('%d'))
    ax.set_xticks([2, 4, 6, 8, 10, 20])

    # Add shot numbers & times
    x0 = 1.035

    tlim_536 = tlim_list[ind_536]
    tlim_538 = tlim_list[ind_538]

    ax.annotate(
        '%i, [%.1f, %.1f] s' % (171536, tlim_536[0], tlim_536[1]),
        (x0, 1.6e-7),
        color=cols[0],
        fontsize=(fontsize - 4))
    ax.annotate(
        '%i, [%.1f, %.1f] s' % (171538, tlim_538[0], tlim_538[1]),
        (x0, 1.15e-7),
        color=cols[1],
        fontsize=(fontsize - 4))

    plt.tight_layout()
    plt.show()
