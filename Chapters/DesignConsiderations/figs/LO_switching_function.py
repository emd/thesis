import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from distinct_colours import get_distinct


# Plotting parameters
fontsize = 16
mpl.rcParams['xtick.labelsize'] = fontsize - 2
mpl.rcParams['ytick.labelsize'] = fontsize - 2
linewidth = 2
cols = get_distinct(2)


class LO(object):
    def __init__(self, Fs=360., tlim=[0, 3], phase_shift=0):
        '''Create instance of local oscillator (LO) object.

        Input parameters:
        -----------------
        Fs - float
            Sampling rate.
            [Fs] = 1 / [T], where T is the period of the LO.

        tlim - array_like, (2,)
            Lower and upper bounds in time.
            [tlim] = [T], where T is the period of the LO.

        phase_shift - float
            Phase shift relative to unshifted (cosine) LO.
            [phase_shift] = rad

        '''
        # Time base
        self.tlim = tlim
        dt = 1. / Fs
        self.t = np.arange(tlim[0], tlim[1] + dt, dt)

        # Signal
        self.x = np.cos((2 * np.pi * self.t) + phase_shift)


if __name__ == '__main__':
    # Generate signals
    LO_I = LO()
    LO_Q = LO(phase_shift=(np.pi / 2))

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    # Plot incident LO
    axes[0].plot(LO_I.t, LO_I.x, c='k', linewidth=linewidth)
    axes[0].axhline(y=0, c='k', linestyle=':', linewidth=linewidth)
    axes[0].set_ylabel('$\mathrm{LO} \, [\mathrm{AU}]$', fontsize=fontsize)

    # Plot switching functions
    axes[1].plot(LO_I.t, np.sign(LO_I.x),
                 c=cols[0], linewidth=linewidth)
    axes[1].plot(LO_Q.t, np.sign(LO_Q.x),
                 c=cols[1], linewidth=linewidth, linestyle='--')
    axes[1].axhline(y=0, c='k', linestyle=':', linewidth=linewidth)
    axes[1].set_ylim([-2, 2])
    axes[1].set_xlabel('$t \, [2 \pi / \Delta \omega_0]$', fontsize=fontsize)
    # axes[1].set_ylabel(
    #     '$\mathrm{sign} \, [\mathrm{unitless}]$', fontsize=fontsize)
    axes[1].set_ylabel(
        r'$\mathrm{switching} \;\; \mathrm{function}$', fontsize=fontsize)

    # Manage tick marks
    xticks = np.arange(np.int(LO_I.tlim[0]), np.int(LO_I.tlim[1]) + 1, 1)
    yticks = np.array([-1, 0, 1])

    axes[0].set_xticks(xticks)
    axes[0].set_yticks(yticks)
    axes[1].set_xticks(xticks)
    axes[1].set_yticks(yticks)

    # Legend
    labels = [
        '$\mathrm{sgn}(\mathrm{LO}_{0})$',
        '$\mathrm{sgn}(\mathrm{LO}_{\pi / 2})$'
    ]

    plt.legend(labels, loc='lower right', fontsize=fontsize)

    plt.show()
