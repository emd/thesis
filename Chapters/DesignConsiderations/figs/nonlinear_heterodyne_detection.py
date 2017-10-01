import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from distinct_colours import get_distinct
from filter import fir

import random_data as rd


# Plotting parameters
fontsize = 16
mpl.rcParams['xtick.labelsize'] = fontsize - 2
mpl.rcParams['ytick.labelsize'] = fontsize - 2
linewidth = 2


class Detector(object):
    def __init__(self, Isat=1.):
        '''Create instance of the `Detector` class.

        Input parameters:
        -----------------
        Isat - float
            The linear saturation intensity of the detector.
            The detector response deviates substantially from
            linearity when the incident optical intensity
            exceeds `Isat`.

            Specifically, `Isat` is the 1 dB saturation point
            of the detector such that the *power* in the
            electrical signal resulting from incident
            optical intensity `Isat` is 1 dB below
            the power that would be expected if the detector
            was completely linear.

            [Isat] = AU

        '''
        self.Isat = np.float(Isat)

    def getLinearResponse(self, I):
        'Get perfectly linear response w/ no saturation.'
        if not np.alltrue(I >= 0):
            raise ValueError('Intensity `I` must be >= 0.')

        return I.copy()

    def getHardResponse(self, I):
        'Get response with hard saturation for I > Isat.'
        if not np.alltrue(I >= 0):
            raise ValueError('Intensity `I` must be >= 0.')

        # Clip if incident optical intensity exceeds saturation
        sat_ind = np.where(I > self.Isat)[0]
        response = I.copy()
        response[sat_ind] = self.Isat

        return response

    def getArctangentResponse(self, I):
        '''Get detector signal in response to incident optical intensity `I`,
        including simple arctangent model of detector saturation.

        '''
        if not np.alltrue(I >= 0):
            raise ValueError('Intensity `I` must be >= 0.')

        def linearity_deviation(x, c):
            return (np.arctan(c * x) / c) / x

        # This multiplicative factor forces the 1 dB saturation point
        # in the output signal's electrical power for the arctangent response
        # to occur when I = Isat
        c = curve_fit(linearity_deviation, [1], [10 ** (-1. / 20)], p0=1)[0][0]

        return self.Isat * np.arctan(c * (I / self.Isat)) / c

    def plotArctangentLinearityDeviation(self, resolution=0.01):
        x = np.arange(resolution, 10 * self.Isat + resolution, resolution)

        deviation = self.getArctangentResponse(x) / self.getLinearResponse(x)

        plt.figure()
        plt.plot(x, deviation)
        plt.axhline(10 ** (-1. / 20), c='k', linestyle='--')
        plt.axvline(self.Isat, c='k', linestyle='--')
        plt.xlabel('intensity')
        plt.ylabel('linearity deviation')
        plt.show()

        return


class Phase(object):
    def __init__(self, Fs, T, Nreal_per_ens,
                 broadband={'Gxx1': 1e-10, 'fc': 100e3, 'pole': 1},
                 coherent={'ph0': 1e-3, 'f0': 200e3}):
        # Generate broadband signal
        ph = rd.signals.RandomSignal(
            Fs, T, fc=broadband['fc'], pole=broadband['pole'])

        self.Fs = Fs
        self.T = ph.t[-1] - ph.t[0]

        # Normalize broadband to appropriate average spectral density
        asd = rd.spectra.AutoSpectralDensity(
            ph.x, Fs=Fs, t0=0,
            Tens=self.T, Nreal_per_ens=Nreal_per_ens)

        ph.x *= np.sqrt(broadband['Gxx1'] / asd.Gxx[1])

        # Generate coherent signal and add to broadband
        ph.x += coherent['ph0'] * np.cos(2 * np.pi * coherent['f0'] * ph.t)

        self.ph = ph.x


class HeterodyneSignals(object):
    def __init__(self, ph, fhet=30e6, Imax=1, saturation='hard'):
        '''Create an instance of `HeterodyneSignals` class.

        Input parameters:
        -----------------
        ph - :py:class:`Phase <detector_linearity.Phase>` instance

        fhet - float
            Heterodyne frequency.
            [fhet] = arbitrary units

        Imax - float
            The maximum intensity of the incident optical signal, where
            Imax = 1 corresponds to the detector's saturation intensity.
            [Imax] = [D.Isat], where D is an instance of
                      :py:class:`Detector <detector_linearity.Detector>`

        saturation - None or string
            Specify detector saturation method. Valid options are
            {None, 'hard', 'atan'}, which are discussed in
            :py:class:`Detector <detector_linearity.Detector>`

        '''
        self.fhet = fhet
        self.t0 = 0
        self.Imax = Imax

        # Set phase modulation
        self.Fs = ph.Fs
        self.T = ph.T
        self.ph = ph.ph

        # Create detector instance
        self.detector = Detector()
        self.saturation = saturation

        # Generate heterodyne signals
        self.LO = np.cos(2 * np.pi * fhet * self.t())
        self.IF = self.getIF()

        # Demodulate heterodyne signals to get measured phase
        self.ph_m = self.getDemodulatedPhase()

    def t(self):
        return np.arange(self.t0, self.T, 1. / self.Fs)

    def getIF(self):
        # Build up intensity piece by piece
        I = np.cos((2 * np.pi * self.fhet * self.t()) + self.ph)
        I += 1
        I *= (0.5 * self.Imax)

        if self.saturation is None:
            return self.detector.getLinearResponse(I)
        elif str.lower(self.saturation) == 'hard':
            return self.detector.getHardResponse(I)
        elif str.lower(self.saturation) == 'atan':
            return self.detector.getArctangentResponse(I)
        else:
            raise ValueError("`saturation` must be in {None, 'hard', 'atan'}")

    def getDemodulatedPhase(self):
        # Subtract mean from IF signal such that it is AC coupled
        IF = self.IF - np.mean(self.IF)

        # # Bandpass filter IF
        # bpf = fir.NER(
        #     0.01,
        #     [0.9 * self.fhet, 1.1 * self.fhet],
        #     0.1 * self.fhet,
        #     self.Fs,
        #     pass_zero=False)

        # IF = bpf.applyTo(IF)

        # Compute analytic representation of signals and extract phase
        ph_IF = np.unwrap(np.angle(hilbert(IF)))
        ph_LO = np.unwrap(np.angle(hilbert(self.LO)))

        return ph_IF - ph_LO


def plot_arctangent_deviation_from_linearity():
    det1 = Detector(Isat=1)
    det10 = Detector(Isat=10)
    det01 = Detector(Isat=0.1)

    det1.plotArctangentLinearityDeviation()
    det10.plotArctangentLinearityDeviation()
    det01.plotArctangentLinearityDeviation()

    return


def plot_detector_response_models(
        Isat=1, saturations=[None, 'atan', 'hard'],
        labels=['no saturation, linear', 'arctangent saturation', 'hard saturation'],
        ax=None, cols=get_distinct(3)):
    det = Detector(Isat=Isat)

    I = np.logspace(
        np.log10(Isat) - 1,
        np.log10(Isat) + 1,
        num=101)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    for i, saturation in enumerate(saturations):
        if saturation is None:
            ax.loglog(I, det.getLinearResponse(I),
                      c=cols[i], linewidth=linewidth)
        elif saturation == 'atan':
            ax.loglog(I, det.getArctangentResponse(I),
                      c=cols[i], linewidth=linewidth)
        elif saturation == 'hard':
            ax.loglog(I, det.getHardResponse(I),
                      c=cols[i], linewidth=linewidth)

    ax.set_xlabel(r'$I \; [I_{\mathrm{sat}}]$', fontsize=fontsize)
    ax.set_ylabel(r'$V(I) \, [V_{\mathrm{sat}}]$', fontsize=fontsize)
    ax.set_ylim([I[0], I[-1]])
    ax.set_title('detector saturation model')

    ax.legend(labels, loc='upper left')

    plt.show()

    return


def plot_example_waveforms(sigs, ax=None, col=get_distinct(1)):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    t = sigs.t() * 1e9
    tlo = 0
    thi = 100
    tind = np.where(np.logical_and(t >= tlo, t <= thi))[0]

    ax.plot(t[tind], sigs.IF[tind], c=col, linewidth=linewidth)

    ax.set_ylim([0, 10])
    ax.set_xlim([tlo, thi])
    ax.set_xlabel('$t \; [\mathrm{ns}]$', fontsize=fontsize)
    ax.set_ylabel(r'$V(t) \; [V_{\mathrm{sat}}]$', fontsize=fontsize)
    ax.set_title('detector IF waveforms')

    plt.show()

    return


def plot_IF_spectra(sigs, Tens, Nreal_per_ens,
                    flim=[0, 80e6], ax=None, col=get_distinct(1)):
    flim = np.asarray(flim)

    asd = rd.spectra.AutoSpectralDensity(
        sigs.IF, Fs=sigs.Fs, t0=sigs.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    Hz_per_MHz = 1e6
    flim /= Hz_per_MHz
    asd.f /= Hz_per_MHz
    asd.Gxx *= Hz_per_MHz

    find = np.where(np.logical_and(
        asd.f > flim[0],
        asd.f <= flim[1]))[0]

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.semilogy(
        asd.f[find], np.mean(asd.Gxx[find, :], axis=-1),
        c=col, linewidth=linewidth)

    ax.set_xlabel('$f \; [\mathrm{MHz}]$', fontsize=fontsize)
    ax.set_ylabel('$G_{VV}(f) \; [V_{\mathrm{sat}}^2 / \mathrm{MHz}]$',
                  fontsize=fontsize)
    ax.set_title('IF spectra')
    ax.set_xlim(flim)
    ax.set_yticks([1e-8, 1e-4, 1e0, 1e4])

    plt.show()

    return


def plot_baseband_spectra(sigs, Tens, Nreal_per_ens,
                          flim=[10e6, 20e6], ax=None, col=get_distinct(1)):
    flim = np.asarray(flim)

    asd = rd.spectra.AutoSpectralDensity(
        sigs.ph_m, Fs=sigs.Fs, t0=sigs.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    Hz_per_MHz = 1e6
    flim /= Hz_per_MHz
    asd.f /= Hz_per_MHz
    asd.Gxx *= Hz_per_MHz

    find = np.where(np.logical_and(
        asd.f > flim[0],
        asd.f <= flim[1]))[0]

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.semilogy(
        asd.f[find], np.mean(asd.Gxx[find, :], axis=-1),
        c=col, linewidth=linewidth)

    ax.set_xlabel('$f \; [\mathrm{MHz}]$', fontsize=fontsize)
    ax.set_ylabel('$G_{\phi\phi}(f) \; [{\mathrm{rad}}^2 / \mathrm{MHz}]$',
                  fontsize=fontsize)
    ax.set_title('demodulated spectra')
    ax.set_xlim(flim)

    plt.show()

    return


if __name__ == '__main__':
    fig, axes = plt.subplots(4, 1, figsize=(9, 12))
    cols = get_distinct(3)

    # Macroscopic properties of heterodyne signal
    fhet = 30e6
    Imax = 1

    # Properties of phase fluctuations and spectral estimation parameters
    Fs = 100 * fhet
    T = 1e-3
    Nreal_per_ens = 100
    broadband = {'Gxx1': 1e-10, 'fc': 250e3, 'pole': 1}
    coherent = {'ph0': 1e-2, 'f0': 16e6}
    ph = Phase(Fs, T, Nreal_per_ens, broadband=broadband, coherent=coherent)
    Tens = ph.T

    # Saturation information
    saturation_type = [None, 'atan', 'hard']
    saturation_labels = [
        'no saturation, linear',
        'arctangent saturation',
        'hard saturation'
    ]

    plot_detector_response_models(
        Isat=1., saturations=saturation_type, labels=saturation_labels,
        ax=axes[0], cols=cols)

    for sind, saturation in enumerate(saturation_type):
        col = cols[sind]

        # Generate heterodyne signals
        sigs = HeterodyneSignals(
            ph, fhet=fhet, Imax=Imax, saturation=saturation)

        # Plot example waveforms
        plot_example_waveforms(sigs, ax=axes[1], col=col)

        # Plot IF spectra
        plot_IF_spectra(sigs, Tens, Nreal_per_ens,
                        flim=[0, 80e6], ax=axes[2], col=col)

        # Plot baseband spectra
        plot_baseband_spectra(sigs, Tens, Nreal_per_ens,
                              flim=[10e6, 20e6], ax=axes[3], col=col)

    plt.tight_layout()

    plt.show()
