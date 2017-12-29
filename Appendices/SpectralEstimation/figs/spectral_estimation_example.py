import numpy as np
import matplotlib.pyplot as plt

import mitpci
import random_data as rd


# Shot parameters
shot = 171505
tlim_load = [2.95, 3.10]
tau = 3.4531906947457383e-08

# Spectral-estimation parameters
tlim_spec = [2.97, 3.06]
Nreal = 200
p = 4
Nk = 1000

# Plotting parameters
figsize = (7, 9)
flim = [10, 2000]
vlim = [1e-10, 1e-4]


if __name__ == '__main__':
    # Load data
    channels = np.arange(3, 17)
    Ph_pci = mitpci.pci.Phase(
        shot,
        channels,
        tlim=tlim_load,
        tau=tau)

    # Spectral estimate
    corr = mitpci.pci.ComplexCorrelationFunction(
        Ph_pci,
        tlim=tlim_spec,
        Nreal_per_ens=Nreal)

    # Plot normalized correlation function
    corr.plotNormalizedCorrelationFunction()
    plt.annotate(
        '%i, [%.2f s, %.2f s]' % (shot, tlim_spec[0], tlim_spec[1]),
        (7, 1900),
        fontsize=10)

    # Compute spectra via Fourier and Burg methods
    asd2d_fourier = mitpci.pci.TwoDimensionalAutoSpectralDensity(
        corr,
        spatial_method='fourier')
    asd2d_burg = mitpci.pci.TwoDimensionalAutoSpectralDensity(
        corr,
        spatial_method='burg',
        burg_params={'p': p, 'Nk': Nk})

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=figsize)
    asd2d_fourier.plotSpectralDensity(
        flim=flim,
        vlim=vlim,
        xlabel='',
        title='Fourier in space',
        ax=ax[0])
    asd2d_burg.plotSpectralDensity(
        flim=flim,
        vlim=vlim,
        title='p = %i Burg AR in space' % p,
        ax=ax[1])

    labels = ['a', 'b']
    for i in np.arange(2):
        ax[i].annotate(
            '(%s)' % labels[i],
            (-25, 1850),
            fontsize=16,
            color='white')
        ax[i].annotate(
            '%i, [%.2f s, %.2f s]' % (shot, tlim_spec[0], tlim_spec[1]),
            (6, 1900),
            fontsize=10,
            color='white')

    plt.tight_layout()
    plt.show()
