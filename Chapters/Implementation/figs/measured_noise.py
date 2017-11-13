import numpy as np
import matplotlib.pyplot as plt
import random_data as rd
import mitpci
import filters


# Spectral-estimation parameters (sets frequency resolution)
Tens = 0.05         # [Tens] = s
Nreal_per_ens = 50

# FIR, zero-delay, high-pass filter
hpf = filters.fir.Kaiser(
    -120, 2.5e3, 5e3,
    pass_zero=False, Fs=4e6)

# Conversion factors
Hz_per_kHz = 1e3


def quantization_noise_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens, Vpp=8.):
    '''Compute one-sided autospectral density of quantization noise.

    Parameters:
    -----------
    Tens - float
        The temporal duration of an ensemble.
        [Tens] = s

    Nreal_per_ens - int
        The number of realizations to split the ensemble into
        (defaults to 50% overlap between Hanning windowed
        realizations).

    Vpp - float
        The dynamic range of the digitizer.
        [Vpp] = V

    Returns:
    --------
    (f, Gxx, shot) - tuple, where

    f - array_like, (`N`,)
        Frequencies in autospectral-density estimate.
        [f] = kHz

    Gxx - array_like, (`N`,)
        The autospectral-density estimate at each frequency `f`.
        [Gxx] = rad^2 / kHz

    shot - int
        The PCI shot number.

    Notes:
    ------
    1034 & 1035 are also good tests of the actual quantization noise,
    but they were digitized at Fs = 10 MSPS rather than the usual 4 MSPS.

    '''
    # Load I&Q data
    L = mitpci.interferometer.Lissajous(
        1069, fit=False, compensate=False)

    # Spectral calculations (I & Q are both ~0, as there
    # is not input to the digitizer, so need to compute
    # the spectra in this manner rather than the usual
    # approach to compute the phase first). Use linear
    # de-trending (rather than usual high-pass filter,
    # which then requires us to specify the "valid" range,
    # corresponding start times, etc.) to remove DC component.
    asd = rd.spectra.AutoSpectralDensity(
        L.I.x, Fs=L.I.Fs, t0=L.I.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens,
        detrend='linear')
    asd.Gxx += (rd.spectra.AutoSpectralDensity(
        L.Q.x, Fs=L.I.Fs, t0=L.I.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens,
        detrend='linear')).Gxx

    # Convert form units of (V^2 / Hz) to (rad^2 / Hz)
    norm = (0.5 * Vpp) ** 2
    asd.Gxx /= norm

    # Convert from Hz to kHz
    asd.f /= Hz_per_kHz
    asd.Gxx *= Hz_per_kHz

    return asd.f, np.mean(asd.Gxx, axis=-1), L.shot


def self_demodulation_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens,
        compensate_rolloff=True):
    '''Compute one-sided autospectral density from self-demodulation.

    Parameters:
    -----------
    Tens - float
        The temporal duration of an ensemble.
        [Tens] = s

    Nreal_per_ens - int
        The number of realizations to split the ensemble into
        (defaults to 50% overlap between Hanning windowed
        realizations).

    compensate_rolloff - bool
        If True, attempt to compensate for old audio-amp rolloff
        for f > 300 kHz, which could distract from the point
        of the plot.

    Returns:
    --------
    (f, Gxx, shot) - tuple, where

    f - array_like, (`N`,)
        Frequencies in autospectral-density estimate.
        [f] = kHz

    Gxx - array_like, (`N`,)
        The autospectral-density estimate at each frequency `f`.
        [Gxx] = rad^2 / kHz

    shot - int
        The PCI shot number.

    Notes:
    ------
    Phase noise from self demodulation with all electronics (tau = 2.5 us).
    (OCXO, ENI RF amplifier, demodulator, audio amps, & digitizer;
    the only complication is that this is with the *old* audio amps,
    which are bandwidth limited to ~300 kHz, after which they roll off;
    try to apply compensation for this roll off so as not to distract).

    '''
    # Load I&Q data
    L = mitpci.interferometer.Lissajous(
        1219, fit=False, compensate=False)

    # Compute phase
    Ph = mitpci.interferometer.Phase(L, filt=hpf)

    # Spectral calculations
    asd = rd.spectra.AutoSpectralDensity(
        Ph.x, Fs=Ph.Fs, t0=Ph.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Convert from Hz to kHz
    asd.f /= Hz_per_kHz
    asd.Gxx *= Hz_per_kHz

    Gxx = np.mean(asd.Gxx, axis=-1)

    # Rolloff compensation; values from measurements
    if compensate_rolloff:
        alpha = 1.5  # Not sure why it's not integer...
        fc = 450.    # [fc] = kHz
        Gxx *= (1 + ((asd.f / fc) ** alpha))

    return asd.f, Gxx, L.shot


def full_system_calibration_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens):
    '''Compute one-sided autospectral density from full system.

    Parameters:
    -----------
    Tens - float
        The temporal duration of an ensemble.
        [Tens] = s

    Nreal_per_ens - int
        The number of realizations to split the ensemble into
        (defaults to 50% overlap between Hanning windowed
        realizations).

    Returns:
    --------
    (f, Gxx, shot) - tuple, where

    f - array_like, (`N`,)
        Frequencies in autospectral-density estimate.
        [f] = kHz

    Gxx - array_like, (`N`,)
        The autospectral-density estimate at each frequency `f`.
        [Gxx] = rad^2 / kHz

    shot - int
        The PCI shot number.

    '''
    # Load I&Q data (it is fine to compensate ellipticity
    # even if I&Q Lissajous figure does not make a full
    # 2 * pi circuit)
    L = mitpci.interferometer.Lissajous(1249)

    # Compute phase
    Ph = mitpci.interferometer.Phase(L, filt=hpf)

    # Spectral calculations
    asd = rd.spectra.AutoSpectralDensity(
        Ph.x, Fs=Ph.Fs, t0=Ph.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Convert from Hz to kHz
    asd.f /= Hz_per_kHz
    asd.Gxx *= Hz_per_kHz

    Gxx = np.mean(asd.Gxx, axis=-1)

    # ... but also remove sound-wave contributions to low-f spectrum
    find = np.where(asd.f <= 30)[0]
    tind = np.where(np.logical_and(
        asd.t >= 0.25,
        asd.t <= 0.75))[0]
    Gxx[find] = np.mean(asd.Gxx[:, tind][find, :], axis=-1)

    return asd.f, Gxx, L.shot


def Lmode_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens):
    '''Compute one-sided autospectral density from typical L-mode.

    Parameters:
    -----------
    Tens - float
        The temporal duration of an ensemble.
        [Tens] = s

    Nreal_per_ens - int
        The number of realizations to split the ensemble into
        (defaults to 50% overlap between Hanning windowed
        realizations).

    Returns:
    --------
    (f, Gxx, shot) - tuple, where

    f - array_like, (`N`,)
        Frequencies in autospectral-density estimate.
        [f] = kHz

    Gxx - array_like, (`N`,)
        The autospectral-density estimate at each frequency `f`.
        [Gxx] = rad^2 / kHz

    shot - int
        The DIII-D shot number.

    '''
    L = mitpci.interferometer.Lissajous(170864, tlim=[0.9, 1.0])

    # Compute phase
    Ph = mitpci.interferometer.Phase(L, filt=hpf)

    # Spectral calculations
    asd = rd.spectra.AutoSpectralDensity(
        Ph.x, Fs=Ph.Fs, t0=Ph.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Convert from Hz to kHz
    asd.f /= Hz_per_kHz
    asd.Gxx *= Hz_per_kHz

    Gxx = np.mean(asd.Gxx, axis=-1)

    return asd.f, Gxx, L.shot


def ELM_free_Hmode_spectral_density(
        Tens=Tens, Nreal_per_ens=Nreal_per_ens):
    '''Compute one-sided autospectral density from typical ELM-free H-mode.

    Parameters:
    -----------
    Tens - float
        The temporal duration of an ensemble.
        [Tens] = s

    Nreal_per_ens - int
        The number of realizations to split the ensemble into
        (defaults to 50% overlap between Hanning windowed
        realizations).

    Returns:
    --------
    (f, Gxx, shot) - tuple, where

    f - array_like, (`N`,)
        Frequencies in autospectral-density estimate.
        [f] = kHz

    Gxx - array_like, (`N`,)
        The autospectral-density estimate at each frequency `f`.
        [Gxx] = rad^2 / kHz

    shot - int
        The DIII-D shot number.

    '''
    L = mitpci.interferometer.Lissajous(170864, tlim=[1.03, 1.11])

    # Compute phase
    Ph = mitpci.interferometer.Phase(L, filt=hpf)

    # Spectral calculations
    asd = rd.spectra.AutoSpectralDensity(
        Ph.x, Fs=Ph.Fs, t0=Ph.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Convert from Hz to kHz
    asd.f /= Hz_per_kHz
    asd.Gxx *= Hz_per_kHz

    Gxx = np.mean(asd.Gxx, axis=-1)

    return asd.f, Gxx, L.shot
