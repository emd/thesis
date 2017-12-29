import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct


rhos = np.arange(0.3, 0.95, 0.05)

# Plotting parameters
fontsize = 15
cols = get_distinct(2)


def load_data(shot, time, rho, sat_rule=1):
    '''Load TGLF-predicted density spectra.

    Input parameters:
    -----------------
    shot - int
        DIII-D shot number.

    time - int
        Time in shot.
        [time] = ms

    rho - float
        The radial coordinate rho for analysis.
        [rho] = unitless

    Returns:
    --------
    data - dict with keys ['density_spec', 'ky'] and
        values corresponding to `N`-length arrays
        of the TGLF-predicted electron-density spectrum
        (density_spec) as a function of wavenumber (ky)
        at radius `rho`.

    '''
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

    # Package data
    data = {
        'density_spec': density_spec,
        'ky': ky
    }

    return data


def plot_data(shot, time, rho, sat_rule=1,
              ax=None, color=cols[0],
              xlim=[0.1, 30], # ylim=[1e-2, 1e1],
              linewidth=2, linestyle='-',
              marker='o', markersize=10, fillstyle='full',
              xlabel=None, ylabel=None, fontsize=16,
              label=None, annotate_rho=False):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    data = load_data(shot, time, rho, sat_rule=sat_rule)

    ky = data['ky']
    density_spec = data['density_spec']

    if fillstyle == 'full':
        markeredgewidth = 0
    else:
        markeredgewidth = linewidth

    ax.loglog(
        ky,
        density_spec,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        fillstyle=fillstyle,
        markeredgewidth=markeredgewidth,
        label=label)

    if xlabel is None:
        ax.set_xlabel(
            r'$\mathregular{k_y \rho_s}$',
            fontsize=fontsize)
    else:
        ax.set_xlabel(
            xlabel,
            fontsize=fontsize)

    if ylabel is None:
        ax.set_ylabel(
            r'$\mathregular{S(k_y) \; [a.u.]}$',
            fontsize=fontsize)
    else:
        ax.set_ylabel(
            ylabel,
            fontsize=fontsize)

    ax.set_xlim(xlim)

    if annotate_rho:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.annotate(
            r'$\mathregular{\rho = %.2f}$' % rho,
            (1.05 * xlim[0], 1.2 * ylim[0]),
            color='k',
            fontsize=fontsize)

    plt.show()

    return ax


if __name__ == '__main__':
    for rho in rhos:
        fig, ax = plt.subplots(1, 1)

        for sat_rule in [0, 1]:
            if sat_rule == 0:
                fillstyle = 'none'
                label05 = r'$\mathregular{\rho_{ECH} = 0.5, standard}$'
                label08 = r'$\mathregular{\rho_{ECH} = 0.8, standard}$'
                annotate_rho = True
            else:
                fillstyle = 'full'
                label05 = r'$\mathregular{\rho_{ECH} = 0.5, multiscale}$'
                label08 = r'$\mathregular{\rho_{ECH} = 0.8, multiscale}$'
                annotate_rho = False

            ax = plot_data(
                171536,
                2750,
                rho,
                sat_rule=sat_rule,
                ax=ax,
                color=cols[0],
                marker='o',
                fillstyle=fillstyle,
                label=label05,
                annotate_rho=annotate_rho)
            ax = plot_data(
                171538,
                2200,
                rho,
                sat_rule=sat_rule,
                ax=ax,
                color=cols[1],
                marker='s',
                fillstyle=fillstyle,
                label=label08)

        ax.legend(
            loc='upper right',
            fontsize=fontsize)

        fname = 'density_spectra_rho0%i.pdf' % np.int(np.round(100 * rho))
        plt.savefig(fname)
        plt.close()
