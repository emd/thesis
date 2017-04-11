import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
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
        self.Isat = Isat

    def getLinearResponse(self, I):
        'Get perfectly linear response w/ no saturation.'
        if not np.alltrue(I >= 0):
            raise ValueError('Intensity `I` must be >= 0.')

        return I / self.Isat

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

        return np.arctan(c * (I / self.Isat)) / c

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
    def __init__(self, fhet=30e6, T=1e-2):
        '''Create an instance of `HeterodyneSignals` class.

        Input parameters:
        -----------------
        fhet - float
            Heterodyne frequency.
            [fhet] = arbitrary units

        T - float
            Signal length.
            [T] = 1 / [fhet]

        '''
        self.Fs = 10 * fhet
        self.T = T
        self.t0 = 0

        # Set phase modulation
        self.ph = self.setPhaseModulation(ph0=1e-3, f0=200e3)

        # Generate heterodyne signals
        self.LO = np.cos(2 * np.pi * fhet * self.t())
        self.IF = np.cos((2 * np.pi * fhet * self.t()) + self.ph)

        # Demodulate heterodyne signals to get measured phase
        self.ph_m = self.getDemodulatedPhase()

    def t(self):
        return np.arange(self.t0, self.T, 1. / self.Fs)

    def setPhaseModulation(self, ph0=1e-3, f0=200e3):
        return ph0 * np.cos(2 * np.pi * f0 * self.t())

    def getDemodulatedPhase(self):
        # Compute analytic representation of signals and extract phase
        ph_IF = np.unwrap(np.angle(hilbert(self.IF)))
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


def plot_detector_response_models():
    cols = get_distinct(3)

    det = Detector(Isat=1)
    I = np.logspace(-1, 1, num=101)

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


if __name__ == '__main__':
    # plot_arctangent_deviation_from_linearity()
    # plot_detector_response_models()
    # plot_effective_waveforms()

    sigs = HeterodyneSignals()

    # Spectral estimation parameters
    Tens = 5e-3
    Nreal_per_ens = 10

    asd = rd.spectra.AutoSpectralDensity(
        sigs.ph, Fs=sigs.Fs, t0=sigs.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    asd_m = rd.spectra.AutoSpectralDensity(
        sigs.ph_m, Fs=sigs.Fs, t0=sigs.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    plt.figure()
    plt.loglog(asd.f, np.mean(asd.Gxx, axis=-1))
    plt.loglog(asd_m.f, np.mean(asd_m.Gxx, axis=-1))
    plt.show()
