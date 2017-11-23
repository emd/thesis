import numpy as np
import matplotlib.pyplot as plt

from power_distribution import (
    reference_beam_power, probe_beam_power,
    eta_R_design, eta_P_design,
    wr, wp,
    s, gaussian_integral_over_square)


# Convert lengths from mm to cm
s /= 10
wr /= 10
wp /= 10

# Detector area
A = s ** 2

# Total beam powers (in W) at detector location
Pr_tot = 1e-3 * reference_beam_power(eta_R_design)
Pp_tot = 1e-3 * probe_beam_power(eta_R_design, eta_P_design)

# Beam powers impinging on square detector element
Pr = Pr_tot * gaussian_integral_over_square(s, wr)
Pp = Pp_tot * gaussian_integral_over_square(s, wp)

# LO parameters
Lf_LO = -165.    # [Lf_LO] = dBc / Hz
tau_LO = 2.5e-6  # [tau_LO] = s

# Digitizer parameters
Nb = 14
Fs = 4e6         # [Fs] = samples / s

# Plotting range
fmin = 10        # [fmin] = kHz
Nf = 1000

# Plotting parameters
fontsize = 12

# Conversion factors
Hz_per_kHz = 1e3


def detector_noise_spectral_density(Dstar=5.35e7, A=A, Pr=Pr, Pp=Pp, Nf=1):
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
        [G] = rad^2 / kHz

    '''
    Gxx = np.ones(Nf) * A / (Pr * Pp * (Dstar ** 2))

    # Convert from rad^2 / Hz -> rad^2 / kHz
    Gxx *= Hz_per_kHz

    return Gxx


def optical_shot_noise_spectral_density(
        Pr=Pr, Pp=Pp, wavelength=10.6e-6, Nf=1):
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
        [G] = rad^2 / kHz

    '''
    h = 6.626e-34  # Planck constant, [h] = J * s
    c = 3e8        # speed of light, [c] = m / w
    photon_energy = h * c / wavelength

    Gxx = 2 * np.ones(Nf) * photon_energy * ((Pr + Pp) / (Pr * Pp))

    # Convert from rad^2 / Hz -> rad^2 / kHz
    Gxx *= Hz_per_kHz

    return Gxx


def LO_instrumental_phase_noise_spectral_density(f, Lf=Lf_LO, tau=tau_LO):
    '''Get one-sided autospectral density of instrumental phase noise
    attributable to injection of local-oscillator (LO) phase noise.

    Parameters:
    -----------
    f - array_like, (`Nf`,)
        The frequencies for which the autospectral density
        should be computed. Note that `f` should not exceed
        the range for which `Lf` is valid.
        [f] = kHz

    Lf - float
        The LO phase noise over the frequency range of interest.
        [Lf] = dBc / Hz

    tau - float
        The AOM coupling delay.
        [tau] = s

    Returns:
    --------
    G - array_like, (`Nf`,)
        The one-sided autospectral density.
        [G] = rad^2 / kHz

    '''
    # Convert from kHz -> Hz (and don't alter original `f` array)
    f = f * Hz_per_kHz

    # Convert from dBc / Hz to rad^2 / Hz
    Lf = 10 ** (Lf / 10.)

    Gxx = 8 * (np.sin(np.pi * f * tau) ** 2) * Lf

    # Convert from rad^2 / Hz -> rad^2 / kHz
    Gxx *= Hz_per_kHz

    return Gxx


def expected_quantization_noise_spectral_density(Nb=Nb, Fs=Fs, Nf=1):
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
        [G] = rad^2 / kHz

    '''
    den = 3 * (2 ** (2 * (Nb - 1))) * Fs
    Gxx = np.ones(Nf) / den

    # Convert from rad^2 / Hz -> rad^2 / kHz
    Gxx *= Hz_per_kHz

    return Gxx
