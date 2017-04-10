import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from distinct_colours import get_distinct


# Plotting parameters
fontsize = 16
mpl.rcParams['xtick.labelsize'] = fontsize - 2
mpl.rcParams['ytick.labelsize'] = fontsize - 2
linewidth = 2
cols = get_distinct(3)
# linestyles = ['-.', '--', '-']
linestyles = ['-', '-', '-']


class Demodulated(object):
    def __init__(self, amplitude_imbalance=0., phase_imbalance=0.,
                 DC_offset_I=0., dBc_3rd=None):
        '''Create instance of demodulated object.

        Input parameters:
        -----------------
        amplitude_imbalance - float
            Ratio of power in quadrature (Q) signal relative to
            power in in-phase (I) signal expressed in dB.
            [amplitude_imbalance] = dB

        phase_imbalance - float
            Phase imbalance between in-phase (I) and quadrature (Q) signals.
            [phase_imbalance] = rad

        DC_offset_I - float
            DC offset as a *fraction* of the in-phase (I) signal amplitude.
            [DC_offset_I] = unitless

        dBc_3rd - float or None
            Ratio of power in 3rd harmonic relative to
            power in fundamental expressed in dB.
            [dBc_3rd] = dBc

        '''
        # Demodulated phase
        # [self.ph] = rad
        self.ph = (2 * np.pi / 360) * np.arange(0, 361, 1)

        # Record demodulator imperfections
        self.amplitude_imbalance = amplitude_imbalance
        self.phase_imbalance = phase_imbalance
        self.DC_offset_I = DC_offset_I
        self.dBc_3rd = dBc_3rd

        # Generate I&Q signals
        self.I = self.getInPhase()
        self.Q = self.getQuadrature()

        # Determine relative error in measured phase fluctuation
        self.relative_error = self.getRelativeError()

    def getInPhase(self):
        # Build up signal iteratively
        I = np.cos(self.ph)

        if self.dBc_3rd is not None:
            # Relative to notation in thesis, we are assuming `I3` < 0
            I += (10 ** (self.dBc_3rd / 20.)) * np.cos(3 * self.ph)

        I += self.DC_offset_I

        return I

    def getQuadrature(self):
        ph = self.ph + self.phase_imbalance

        # Build up signal iteratively
        Q = np.sin(ph)

        if self.dBc_3rd is not None:
            # Relative to notation in thesis, we are assuming `Q3` < 0
            Q -= (10 ** (self.dBc_3rd / 20.)) * np.sin(3 * ph)

        Q *= (10 ** (self.amplitude_imbalance / 20.))

        return Q

    def getMeasuredPhase(self):
        return np.unwrap(np.arctan2(self.Q, self.I))

    def getRelativeError(self):
        dph = self.getMeasuredPhase() - self.ph
        return np.diff(dph) / np.diff(self.ph)


if __name__ == '__main__':
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))

    IQ_lim = [-1.25, 1.25]
    IQ_xticks = np.array([-1, 0, 1])
    IQ_yticks = np.array([-1, 0, 1])

    relative_error_xlim = [0, 2 * np.pi]
    relative_error_xticks = np.array([0, np.pi, 2 * np.pi])
    relative_error_xticklabels = [r'0', r'$\pi$', r'$2 \pi$']
    relative_error_ylim = [-0.5, 0.5]
    relative_error_yticks = [-0.25, 0, 0.25]

    # Effect of DC offset
    DC_offsets = np.array([0., 1e-1, 2e-1])
    labels = []

    for i, DC_offset in enumerate(DC_offsets):
        D = Demodulated(DC_offset_I=DC_offset)

        axes[0, 0].plot(
            D.I, D.Q, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])
        axes[0, 1].plot(
            D.ph[:-1], D.relative_error, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])

        labels.append('%i%%' % np.int(DC_offset * 100))

    axes[0, 0].set_ylabel(r'$Q \; [\mathrm{AU}]$', fontsize=fontsize)
    axes[0, 0].set_title('DC offset')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_xlim(IQ_lim)
    axes[0, 0].set_ylim(IQ_lim)
    axes[0, 0].set_xticks(IQ_xticks)
    axes[0, 0].set_xticklabels([])
    axes[0, 0].set_yticks(IQ_yticks)

    axes[0, 1].set_ylabel(
        r'$\delta\tilde{\phi} / \tilde{\phi}$', fontsize=fontsize)
    axes[0, 1].set_title('Effect of DC offset')
    axes[0, 1].set_xlim(relative_error_xlim)
    axes[0, 1].set_xticks(relative_error_xticks)
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_ylim(relative_error_ylim)
    axes[0, 1].set_yticks(relative_error_yticks)

    axes[0, 1].legend(labels, loc='lower right')

    # Effect of amplitude imbalance
    amplitude_imbalances = np.array([0., -1., -3.])
    labels = []

    for i, amplitude_imbalance in enumerate(amplitude_imbalances):
        D = Demodulated(amplitude_imbalance=amplitude_imbalance)

        axes[1, 0].plot(
            D.I, D.Q, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])
        axes[1, 1].plot(
            D.ph[:-1], D.relative_error, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])

        labels.append('%i dB' % np.int(amplitude_imbalance))

    axes[1, 0].set_ylabel(r'$Q \; [\mathrm{AU}]$', fontsize=fontsize)
    axes[1, 0].set_title('Amplitude imbalance')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_xlim(IQ_lim)
    axes[1, 0].set_ylim(IQ_lim)
    axes[1, 0].set_xticks(IQ_xticks)
    axes[1, 0].set_xticklabels([])
    axes[1, 0].set_yticks(IQ_yticks)

    axes[1, 1].set_ylabel(
        r'$\delta\tilde{\phi} / \tilde{\phi}$', fontsize=fontsize)
    axes[1, 1].set_title('Effect of amplitude imbalance')
    axes[1, 1].set_xlim(relative_error_xlim)
    axes[1, 1].set_xticks(relative_error_xticks)
    axes[1, 1].set_xticklabels([])
    axes[1, 1].set_ylim(relative_error_ylim)
    axes[1, 1].set_yticks(relative_error_yticks)

    axes[1, 1].legend(labels, loc='lower right')

    # Effect of phase imbalance
    phase_imbalances = (np.pi / 180) * np.array([0., 10., 20.])
    labels = []

    for i, phase_imbalance in enumerate(phase_imbalances):
        D = Demodulated(phase_imbalance=phase_imbalance)

        axes[2, 0].plot(
            D.I, D.Q, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])
        axes[2, 1].plot(
            D.ph[:-1], D.relative_error, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])

        labels.append(u'%i\u00b0' % np.int((180 / np.pi) * phase_imbalance))

    axes[2, 0].set_ylabel(r'$Q \; [\mathrm{AU}]$', fontsize=fontsize)
    axes[2, 0].set_title('Phase imbalance')
    axes[2, 0].set_aspect('equal')
    axes[2, 0].set_xlim(IQ_lim)
    axes[2, 0].set_ylim(IQ_lim)
    axes[2, 0].set_xticks(IQ_xticks)
    axes[2, 0].set_xticklabels([])
    axes[2, 0].set_yticks(IQ_yticks)

    axes[2, 1].set_ylabel(
        r'$\delta\tilde{\phi} / \tilde{\phi}$', fontsize=fontsize)
    axes[2, 1].set_title('Effect of phase imbalance')
    axes[2, 1].set_xlim(relative_error_xlim)
    axes[2, 1].set_xticks(relative_error_xticks)
    axes[2, 1].set_xticklabels([])
    axes[2, 1].set_ylim(relative_error_ylim)
    axes[2, 1].set_yticks(relative_error_yticks)

    axes[2, 1].legend(labels, loc='lower right')

    # Effect of 3rd harmonic
    dBc_3rd_array = np.array([None, -30., -20.])
    labels = []

    for i, dBc_3rd in enumerate(dBc_3rd_array):
        D = Demodulated(dBc_3rd=dBc_3rd)

        axes[3, 0].plot(
            D.I, D.Q, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])
        axes[3, 1].plot(
            D.ph[:-1], D.relative_error, linewidth=linewidth,
            linestyle=linestyles[i], c=cols[i])

        if dBc_3rd is None:
            labels.append('None')
        else:
            labels.append('%i dBc' % np.int(dBc_3rd))

    axes[3, 0].set_xlabel(r'$I \; [\mathrm{AU}]$', fontsize=fontsize)
    axes[3, 0].set_ylabel(r'$Q \; [\mathrm{AU}]$', fontsize=fontsize)
    axes[3, 0].set_title('3rd harmonic')
    axes[3, 0].set_aspect('equal')
    axes[3, 0].set_xlim(IQ_lim)
    axes[3, 0].set_ylim(IQ_lim)
    axes[3, 0].set_xticks(IQ_xticks)
    axes[3, 0].set_yticks(IQ_yticks)

    axes[3, 1].set_xlabel(r'$\phi \; [\mathrm{rad}]$', fontsize=fontsize)
    axes[3, 1].set_ylabel(
        r'$\delta\tilde{\phi} / \tilde{\phi}$', fontsize=fontsize)
    axes[3, 1].set_title('Effect of 3rd harmonic')
    axes[3, 1].set_xlim(relative_error_xlim)
    axes[3, 1].set_xticks(relative_error_xticks)
    axes[3, 1].set_xticklabels(relative_error_xticklabels)
    axes[3, 1].set_ylim(relative_error_ylim)
    axes[3, 1].set_yticks(relative_error_yticks)

    axes[3, 1].legend(labels, loc='lower right')

    plt.tight_layout()

    plt.show()
