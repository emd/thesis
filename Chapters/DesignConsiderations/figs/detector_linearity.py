import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from distinct_colours import get_distinct

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
        c = curve_fit(linearity_deviation, 1, 10 ** (-1. / 20), p0=1)[0][0]

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


class HeterodyneSignals(object):
    def __init__(self, fhet=30e6, Fs=10., T=1e-2,
                 broadband={'power': 3e-8, 'fc': 100e3, 'pole': 1},
                 coherent={'ph0': 1e-3, 'f0': 200e3},
                 Imax=1, saturation='hard'):
        '''Create an instance of `HeterodyneSignals` class.

        Input parameters:
        -----------------
        fhet - float
            Heterodyne frequency.
            [fhet] = arbitrary units

        Fs - float
            Sampling rate in units of heterodyne frequency
            [Fs] = fhet

        T - float
            Signal length.
            [T] = 1 / [fhet]

        broadband - dict, with keys:

            power - float
                Total power in broadband signal.
                [power] = rad^2

            fc - float
                Cutoff frequency of broadband signal.
                [fc] = [fhet]

            pole - int
                Number of poles in broadband cutoff.
                [pole] = unitless

        coherent - dict, with keys:

            ph0 - float
                Amplitude of coherent signal.
                [ph0] = rad

            f0 - float
                Frequency of coherent signal.
                [f0] = [fhet]

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
        self.Fs = Fs * self.fhet
        self.t0 = 0
        self.Imax = Imax

        # Set phase modulation
        self.ph, self.T = self.setPhaseModulation(
            T, broadband=broadband, coherent=coherent)

        # Create detector instance
        self.detector = Detector()
        self.saturation = saturation

        # Generate heterodyne signals
        self.LO = np.cos(2 * np.pi * fhet * self.t())
        self.IF = self.getIF()

        # Demodulate heterodyne signals to get measured phase
        self.ph_m = self.getDemodulatedPhase()

    def setPhaseModulation(
            self, T,
            broadband={'power': 1e-7, 'fc': 100e3, 'pole': 1},
            coherent={'ph0': 1e-3, 'f0': 200e3}):
        # Generate broadband signal w/ appropriate total power
        ph = rd.signals.RandomSignal(
            self.Fs, T, fc=broadband['fc'], pole=broadband['pole'])
        ph.x *= (broadband['power'] / ((np.std(ph.x)) ** 2))

        # Generate coherent signal and add to broadband
        ph.x += coherent['ph0'] * np.cos(2 * np.pi * coherent['f0'] * ph.t)

        return ph.x, (ph.t[-1] - ph.t[0])

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


def plot_detector_response_models(Isat=1):
    cols = get_distinct(3)

    det = Detector(Isat=Isat)

    I = np.logspace(
        np.log10(Isat) - 1,
        np.log10(Isat) + 1,
        num=101)

    plt.figure()

    plt.loglog(I, det.getLinearResponse(I),
               c=cols[0], linestyle='-.', linewidth=linewidth)
    plt.loglog(I, det.getArctangentResponse(I),
               c=cols[1], linestyle='--', linewidth=linewidth)
    plt.loglog(I, det.getHardResponse(I),
               c=cols[2], linestyle='-', linewidth=linewidth)

    plt.xlabel(r'$I \; [I_{\mathrm{sat}}]$', fontsize=fontsize)
    plt.ylabel(r'$\mathrm{response} \, [\mathrm{AU}]$', fontsize=fontsize)
    plt.ylim([I[0], I[-1]])

    labels = [
        'no saturation, linear',
        'arctangent saturation',
        'hard saturation'
    ]
    plt.legend(labels, loc='lower right')

    plt.show()

    return


def plot_effective_waveforms():
    cols = get_distinct(2)

    det = Detector(Isat=1)

    t = np.arange(0, 3.01, 0.01)
    Ilin = 0.5 * (1 + np.cos(2 * np.pi * t))
    Isat = 10 * Ilin

    plt.figure()

    # Linear
    plt.plot(t, det.getLinearResponse(Ilin),
             c=cols[0], linestyle='-.', linewidth=linewidth)
    plt.plot(t, det.getHardResponse(Ilin),
             c=cols[0], linestyle='--', linewidth=linewidth)
    plt.plot(t, det.getArctangentResponse(Ilin),
             c=cols[0], linestyle='-', linewidth=linewidth)

    # Saturated
    plt.plot(t, det.getLinearResponse(Isat),
             c=cols[1], linestyle='-.', linewidth=linewidth)
    plt.plot(t, det.getHardResponse(Isat),
             c=cols[1], linestyle='--', linewidth=linewidth)
    plt.plot(t, det.getArctangentResponse(Isat),
             c=cols[1], linestyle='-', linewidth=linewidth)

    plt.xlabel('t [AU]')
    plt.ylabel('effective intensity')

    plt.show()

    return


def plot_effective_waveforms_2():
    fhet = 30e6
    T = 3. / fhet

    Imax_array = np.array([1, 10])
    saturation_types = [None, 'hard', 'atan']

    for Imax in Imax_array:
        for saturation in saturation_types:
            sigs = HeterodyneSignals(
                fhet=fhet, T=T,
                Imax=Imax, saturation=saturation)

            plt.figure()
            plt.plot(sigs.t(), sigs.IF)
            plt.title('Imax / Isat: %0.1f, saturation: %s' %
                      (Imax, saturation))
            plt.show()

    return


if __name__ == '__main__':
    # Ancillary routines:
    # -------------------
    # plot_arctangent_deviation_from_linearity()
    plot_detector_response_models(Isat=1.)
    # plot_effective_waveforms()
    # plot_effective_waveforms_2()

    # Main:
    # -----
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    cols = get_distinct(3)

    fhet = 30e6
    broadband = {'power': 3e-8, 'fc': 100e3, 'pole': 1}
    coherent = {'ph0': 0, 'f0': 200e3}

    Imax_array = np.array([1., 10.])
    saturation_type = [None, 'atan', 'hard']
    saturation_labels = [
        'no saturation, linear',
        'arctangent saturation',
        'hard saturation'
    ]

    for Iind, Imax in enumerate(Imax_array):
        for sind, saturation in enumerate(saturation_type):
            # Plot example high-resolution waveforms:
            # ---------------------------------------
            sigs = HeterodyneSignals(
                fhet=fhet, Fs=100., T=(4. / fhet),
                broadband=broadband, coherent=coherent,
                Imax=Imax, saturation=saturation)

            axes[0, Iind].plot(
                sigs.t() * 1e9, sigs.IF,
                c=cols[sind], linewidth=linewidth)

            # Plot measured baseband spectra:
            # -------------------------------
            sigs = HeterodyneSignals(
                fhet=fhet, Fs=10., T=1e-1,
                broadband=broadband, coherent=coherent,
                Imax=Imax, saturation=saturation)

            # Spectral estimation parameters
            Tens = sigs.T
            Nreal_per_ens = 250

            asd = rd.spectra.AutoSpectralDensity(
                sigs.ph_m, Fs=sigs.Fs, t0=sigs.t0,
                Tens=Tens, Nreal_per_ens=Nreal_per_ens)

            # Plotting parameters
            find = np.where(np.logical_and(
                asd.f >= 10e3,
                asd.f <= 1e6))[0]

            axes[1, Iind].loglog(
                asd.f[find], np.mean(asd.Gxx[find, :], axis=-1),
                c=cols[sind], linewidth=linewidth)

        axes[0, Iind].set_ylim([0, 10])
        axes[0, Iind].set_xlim([0, 100])
        axes[0, Iind].set_xlabel('$t \; [\mathrm{ns}]$', fontsize=fontsize)
        axes[0, Iind].axhline(1, c='k', linestyle='--')

        axes[1, Iind].set_ylim([1e-12, 1e-8])
        axes[1, Iind].set_yticks([1e-12, 1e-10, 1e-8])
        axes[1, Iind].set_xlabel('$f \; [\mathrm{Hz}]$', fontsize=fontsize)

    axes[0, 0].legend(saturation_labels, loc='upper right')

    axes[0, 0].set_ylabel(
        r'$I_{\mathrm{eff}} \; [I_{\mathrm{sat}}]$',
        fontsize=fontsize)
    axes[0, 0].set_title(
        r'$I_{\mathrm{max}} = I_{\mathrm{sat}}$',
        fontsize=fontsize)
    axes[1, 0].set_ylabel(
        r'$G_{\phi\phi}(f) \; [\mathrm{rad}^2 / \mathrm{Hz}]$',
        fontsize=fontsize)
    axes[0, 1].set_title(
        r'$I_{\mathrm{max}} = 10 \; I_{\mathrm{sat}}$',
        fontsize=fontsize)

    axes[0, 1].set_yticklabels([])
    axes[1, 1].set_yticklabels([])

    plt.tight_layout()
    plt.show()
