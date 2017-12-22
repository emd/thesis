import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct


rhos = np.arange(0.3, 0.95, 0.05)

# Plotting parameters
fontsize = 15
cols = get_distinct(2)


def load_data(shot, time, rho):
    '''Load TGLF linear-stability data.

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
    data - dict with keys ['freq1', 'gamma1', 'ky'] and
        values corresponding to `N`-length arrays
        of the TGLF-predicted frequency (freq1) and
        growth rate (gamma1) as a function of wavenumber (ky)
        for the most unstable mode at radius `rho`.

    '''
    d = '%i/%ibis/0%i' % (shot, time, np.int(np.round(100 * rho)))

    # Load data
    fname = '%s/%s.pkl' % (d, 'freq1')
    with open(fname, 'rb') as f:
        freq1 = pickle.load(f)

    fname = '%s/%s.pkl' % (d, 'gamma1')
    with open(fname, 'rb') as f:
        gamma1 = pickle.load(f)

    fname = '%s/%s.pkl' % (d, 'ky')
    with open(fname, 'rb') as f:
        ky = pickle.load(f)

    # Package data
    data = {
        'freq1': freq1,
        'gamma1': gamma1,
        'ky': ky
    }

    return data


def plot_data(shot, time, rho,
              ax=None, color=cols[0],
              xlim=[0.1, 30], ylim=[1e-2, 1e1],
              linewidth=2, linestyle='-',
              marker='o', markersize=10,
              xlabel=None, ylabel=None, fontsize=16,
              label=None, annotate_rho=False):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    data = load_data(shot, time, rho)

    ky = data['ky']
    omega = data['freq1']
    gamma = data['gamma1']

    # Ion modes will be plotted with empty markers, and
    # electron modes will be plotted with full markers
    ionfill = 'none'
    eonfill = 'full'

    ion = np.where(omega < 0)[0]
    eon = np.where(omega > 0)[0]

    ax.loglog(
        ky,
        gamma,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label)
    ax.loglog(
        ky[ion],
        gamma[ion],
        color=color,
        linestyle='none',
        marker=marker,
        markersize=markersize,
        fillstyle=ionfill,
        markeredgewidth=linewidth)
    ax.loglog(
        ky[eon],
        gamma[eon],
        color=color,
        linestyle='none',
        marker=marker,
        markersize=markersize,
        fillstyle=eonfill,
        markeredgewidth=0)

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
            r'$\mathregular{\gamma \; [c_s / a]}$',
            fontsize=fontsize)
    else:
        ax.set_ylabel(
            ylabel,
            fontsize=fontsize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if annotate_rho:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.annotate(
            r'$\mathregular{\rho = %.2f}$' % rho,
            (0.4 * xlim[1], 1.2 * ylim[0]),
            color='k',
            fontsize=fontsize)

    plt.show()

    return ax


if __name__ == '__main__':
    for rho in rhos:
        ax = plot_data(
            171536,
            2750,
            rho,
            color=cols[0],
            marker='o',
            label=r'$\mathregular{\rho_{ECH} = 0.5}$',
            annotate_rho=True)
        ax = plot_data(
            171538,
            2200,
            rho,
            ax=ax,
            color=cols[1],
            marker='s',
            label=r'$\mathregular{\rho_{ECH} = 0.8}$')

        ax.legend(
            loc='upper left',
            fontsize=fontsize)

        fname = 'linear_stability_rho0%i.pdf' % np.int(np.round(100 * rho))
        plt.savefig(fname)

        plt.close()
