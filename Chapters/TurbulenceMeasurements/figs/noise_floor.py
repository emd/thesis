import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random_data as rd
import mitpci


shot = 171521
tlim = [-0.1, 0]
pci_channel = 8

Hz_per_kHz = 1e3

# Spectral-estimation parameters
Tens = 5e-3
Nreal_per_ens = 10


def interferometer_noise_floor_model(f):
    'Return interf. model noise floor [rad^2 / kHz] for frequency `f` [kHz].'
    return 1e-10 * np.ones(len(f))


def pci_noise_floor_model(f):
    'Return PCI model noise floor [rad^2 / kHz] for frequency `f` [kHz].'
    nf = np.zeros(len(f))

    # Noise floor is flat at `nf0` below cutoff frequency `fc`;
    # beyond cutoff frequency, noise floor decreases as a result
    # of detector rolloff (that is, noise floor is attributable
    # to detector noise).
    nf0 = 1e-12
    fc = 500.
    rolloff_exponent = 1  # measured, roughly

    # f <= fc
    find = np.where(f <= fc)[0]
    nf[find] = nf0

    # f > fc
    find = np.where(f > fc)[0]
    nf[find] = nf0 * ((f[find] / fc) ** -rolloff_exponent)

    return nf


if __name__ == '__main__':
    # Load data prior to breakdown to see noise floors
    L = mitpci.interferometer.Lissajous(shot, tlim=tlim)
    Ph_int = mitpci.interferometer.Phase(L)
    Ph_pci = mitpci.pci.Phase(shot, pci_channel, tlim=tlim)

    # Compute spectra
    asd_int = rd.spectra.AutoSpectralDensity(
        Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    asd_pci = rd.spectra.AutoSpectralDensity(
        Ph_pci.x, Fs=Ph_pci.Fs, t0=Ph_pci.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Unit conversions
    asd_int.f /= Hz_per_kHz
    asd_pci.f /= Hz_per_kHz
    asd_int.Gxx *= Hz_per_kHz
    asd_pci.Gxx *= Hz_per_kHz

    # Plot and compare to noise models
    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].loglog(asd_int.f, np.mean(asd_int.Gxx, axis=-1))
    axes[0].loglog(
        asd_int.f,
        interferometer_noise_floor_model(asd_int.f),
        linestyle='--', c='k')
    axes[0].set_title('interferometer')
    axes[0].set_xlabel('f [kHz]')
    axes[0].set_ylabel('noise floor [rad^2 / kHz]')

    axes[1].loglog(asd_pci.f, np.mean(asd_pci.Gxx, axis=-1))
    axes[1].loglog(
        asd_pci.f,
        pci_noise_floor_model(asd_pci.f),
        linestyle='--', c='k')
    axes[1].set_title('PCI')
    axes[1].set_xlabel('f [kHz]')
    axes[1].set_ylabel('noise floor [rad^2 / kHz]')

    plt.show()
