import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from distinct_colours import get_distinct


# Plotting parameters
fontsize = 16
mpl.rcParams['xtick.labelsize'] = fontsize - 2
mpl.rcParams['ytick.labelsize'] = fontsize - 2
linewidth = 2
cols = get_distinct(3)
# linestyles = ['-.', '--', '-']
linestyles = ['-', '-', '-']


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

        # This multiplicative factor forces the 1 dB saturation point
        # in the output signal's electrical power for the arctangent response
        # to occur when I = Isat
        self._c = curve_fit(
            self._getArctangentLinearityDeviation,
            1, 10 ** (-1. / 20), p0=1)[0][0]

    def _getArctangentLinearityDeviation(self, x, c):
        '''Get deviation of arctangent from linear, expressed as
        a ratio of arctangent to linear.

        '''
        return np.arctan(c * x) / (c * x)

    def getLinearResponse(self, I):
        '''Get perfectly linear response w/ no saturation, but
        include appropriate scaling for comparison to arctangent model.

        '''
        if not np.alltrue(I >= 0):
            raise ValueError('Intensity `I` must be >= 0.')

        return self._c * (I / self.Isat)

    def getHardResponse(self, I):
        '''Get response with hard saturation for I > Isat, and
        include appropriate scaling for comparison to arctangent model.

        '''
        if not np.alltrue(I >= 0):
            raise ValueError('Intensity `I` must be >= 0.')

        # Clip if incident optical intensity exceeds saturation
        sat_ind = np.where(I > self.Isat)[0]
        response = I.copy()
        response[sat_ind] = self.Isat

        response *= self._c

        return response

    def getArctangentResponse(self, I):
        '''Get detector signal in response to incident optical intensity `I`,
        including simple arctangent model of detector saturation.

        '''
        if not np.alltrue(I >= 0):
            raise ValueError('Intensity `I` must be >= 0.')

        return np.arctan(self._c * (I / self.Isat))

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


class HeterodyneSignal(object):
    def __init__(self, fhet=30e6):
        pass


class Demodulated(object):
    def __init__(self):
        pass


def plot_detector_response_models():
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

    labels = [
        'no saturation, linear',
        'arctangent saturation',
        'hard saturation'
    ]
    plt.legend(labels, loc='lower right')

    plt.show()

if __name__ == '__main__':
    plot_detector_response_models()
