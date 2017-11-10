import numpy as np
import matplotlib.pyplot as plt

from power_distribution import (
    reference_beam_power, probe_beam_power,
    eta_R_design, eta_P_design,
    wr, wp,
    s, gaussian_integral_over_square)


# Specific detectivity
Dstar = 5e7      # [Dstar] = Jones = cm * Hz^{1/2} / W

# LO parameters
Lf_LO = -165.    # [Lf_LO] = dBc / Hz
tau_LO = 2.5e-6  # [tau_LO] = s

# Digitizer parameters
Nb = 14
Fs = 4e6         # [Fs] = samples / s

# Plotting range
fmin = 10e3      # [fmin] = Hz
Nf = 1000

# Plotting parameters
fontsize = 12


def detector_noise_spectral_density(Dstar, A, Pr, Pp, Nf=1):
    '''Get one-sided autospectral density of demodulated detector noise.

    Parameters:
    -----------
    Dstar - float
        Specific detectivity of detector.
        [Dstar] = Jones = cm * Hz^{1/2} / W

    A - float
        Active area of detector element.
        [A] = cm^2

    Pr - float
        Optical power of reference beam impinging on detector element.
        [Pr] = W

    Pp - float
        Optical power of probe beam impinging on detector element.
        [Pp] = W

    Nf - int
        The number of frequencies for which the autospectral density
        should be returned.

    Returns:
    --------
    G - array_like, (`Nf`,)
        The one-sided autospectral density.
        [G] = rad^2 / Hz

    '''
    return np.ones(Nf) * A / (Pr * Pp * (Dstar ** 2))


def optical_shot_noise_spectral_density(Pr, Pp, wavelength=10.6e-6, Nf=1):
    '''Get one-sided autospectral density of demodulated optical shot noise.

    Parameters:
    -----------
    Pr - float
        Optical power of reference beam impinging on detector element.
        [Pr] = W

    Pp - float
        Optical power of probe beam impinging on detector element.
        [Pp] = W

    wavelength - float
        Laser wavelength.
        [wavelength] = m

    Nf - int
        The number of frequencies for which the autospectral density
        should be returned.

    Returns:
    --------
    G - array_like, (`Nf`,)
        The one-sided autospectral density.
        [G] = rad^2 / Hz

    '''
    h = 6.626e-34  # Planck constant, [h] = J * s
    c = 3e8        # speed of light, [c] = m / w
    photon_energy = h * c / wavelength

    return 2 * np.ones(Nf) * photon_energy * ((Pr + Pp) / (Pr * Pp))


def LO_instrumental_phase_noise_spectral_density(Lf, tau, f):
    '''Get one-sided autospectral density of instrumental phase noise
    attributable to injection of local-oscillator (LO) phase noise.

    Parameters:
    -----------
    Lf - float
        The LO phase noise over the frequency range of interest.
        [Lf] = dBc / Hz

    tau - float
        The AOM coupling delay.
        [tau] = s

    f - array_like, (`Nf`,)
        The frequencies for which the autospectral density
        should be computed. Note that `f` should not exceed
        the range for which `Lf` is valid.
        [f] = Hz

    Returns:
    --------
    G - array_like, (`Nf`,)
        The one-sided autospectral density.
        [G] = rad^2 / Hz

    '''
    # Convert from dBc / Hz to rad^2 / Hz
    Lf = 10 ** (Lf / 10.)

    return 8 * (np.sin(np.pi * f * tau) ** 2) * Lf


def quantization_noise_spectral_density(Nb, Fs, Nf=1):
    '''Get one-sided autospectral density of quantization noise.

    Parameters:
    -----------
    Nb - int
        The bit depth.

    Fs - float
        The sampling rate.
        [Fs] = samples / s

    Nf - int
        The number of frequencies for which the autospectral density
        should be returned.

    Returns:
    --------
    G - array_like, (`Nf`,)
        The one-sided autospectral density.
        [G] = rad^2 / Hz

    '''
    den = 3 * (2 ** (2 * (Nb - 1))) * Fs
    return np.ones(Nf) / den


if __name__ == '__main__':
    # Convert lengths from mm to cm
    s /= 10
    wr /= 10
    wp /= 10

    # Detector area
    A = s ** 2

    # Total beam powers (in W) at detector location
    Pr_tot = 1e-3 * reference_beam_power(eta_R_design)
    Pp_tot = 1e-3 * probe_beam_power(eta_R_design, eta_P_design)
    # print '\nPr_tot: %f mW' % (Pr_tot * 1e3)
    # print 'Pp_tot: %f mW' % (Pp_tot * 1e3)

    # Beam powers impinging on square detector element
    Pr = Pr_tot * gaussian_integral_over_square(s, wr)
    Pp = Pp_tot * gaussian_integral_over_square(s, wp)
    # print '\nPr: %f mW' % (Pr * 1e3)
    # print 'Pp: %f mW' % (Pp * 1e3)

    # Construct computational grid
    f = np.logspace(np.log10(fmin), np.log10(Fs / 2), Nf)

    plt.figure()

    # Demodulated detector noise
    plt.loglog(
        f,
        detector_noise_spectral_density(Dstar, A, Pr, Pp, Nf=len(f)),
        label='detector')

    # Demodulated optical shot noise
    plt.loglog(
        f,
        optical_shot_noise_spectral_density(Pr, Pp, Nf=len(f)),
        label='optical shot noise')

    # LO phase noise (don't plot last point, as it corresponds
    # to a null and totally messes up the plot range)
    plt.loglog(
        f[:-1],
        LO_instrumental_phase_noise_spectral_density(Lf_LO, tau_LO, f)[:-1],
        label='local oscillator, $\\tau = 2.5 \, \mu\mathregular{s}$')

    # Quantization noise
    plt.loglog(
        f,
        quantization_noise_spectral_density(Nb, Fs, Nf=len(f)),
        label='quantization')

    plt.xlabel(
        r'$f \; [\mathregular{Hz}]$',
        fontsize=fontsize)
    plt.ylabel(
        r'$G(f) \; [\mathregular{rad^2 / \, Hz}]$',
        fontsize=fontsize)
    plt.legend(loc='best')

    plt.xlim([f[0], f[-1]])
    plt.show()
