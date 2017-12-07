import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random_data as rd
import mitpci
import magnetics


# Shot parameters
shot = 167341
tlim = [2.0, 2.2]  # [tlim] = s

# Spectral-estimation parameters
Tens = 25e-3         # [Tens] = s
Nreal_per_ens = 10

# Unit conversions
Hz_per_kHz = 1e3

# Plotting parameters
gamma2xy_threshold = 0.5
R2_threshold = 0.9
all_positive = False
flim = [0., 100.]   # [flim] = kHz
figsize = (9, 5)
fontsize = 15
annotation_color = 'r'
annotation_linewidth = 4
annotation_linestyle = '-'


if __name__ == '__main__':
    # Interferometers:
    # ----------------
    # Load PCI interferometer data
    L = mitpci.interferometer.Lissajous(shot, tlim=tlim)
    Ph = mitpci.interferometer.Phase(L)

    # Compute interferometer-measured toroidal mode-number spectrum
    TorCorr = mitpci.interferometer.ToroidalCorrelation(
        Ph,
        Tens=Tens,
        Nreal_per_ens=Nreal_per_ens)

    # Magnetics:
    # ----------
    # Load magnetics data
    torsigs = magnetics.signal.ToroidalSignals(
        shot,
        tlim=tlim)

    # Compute magnetics-measured toroidal mode-number spectrum
    A = rd.array.FittedCrossPhaseArray(
        torsigs.x,
        torsigs.locations,
        Fs=torsigs.Fs,
        t0=torsigs.t0,
        Tens=Tens,
        Nreal_per_ens=Nreal_per_ens)

    # Distinct, color-blind proof colormap
    # (Restrict magnetics mode numbers to those below
    # interferometer Nyquist limit, so use interferometer
    # colormap).
    if all_positive:
        cmap_mag = magnetics.colormap.positive_mode_numbers()[1]
        mag_mode_number_lim = [0, 7]
    else:
        cmap_mag = magnetics.colormap.mixed_sign_mode_numbers()[1]
        mag_mode_number_lim = [-3, 4]

    # Unit conversions:
    # -----------------
    TorCorr.f /= Hz_per_kHz
    TorCorr.df /= Hz_per_kHz
    A.f /= Hz_per_kHz
    A.df /= Hz_per_kHz

    # Plotting:
    # ---------
    # Plot spectrum using "nearest" interpolation
    # to prevent aliasing in saved figured, as described
    # here:
    #
    #   https://github.com/matplotlib/matplotlib/issues/2972/
    #
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)

    xlabel = r'$\mathregular{t \; [s]}$'
    ylabel = r'$\mathregular{f \; [kHz]}$'
    interpolation = 'nearest'

    TorCorr.plotModeNumber(
        gamma2xy_threshold=gamma2xy_threshold,
        all_positive=all_positive,
        tlim=tlim,
        flim=flim,
        xlabel=xlabel,
        ylabel=ylabel,
        title='interferometers',
        fontsize=fontsize,
        interpolation=interpolation,
        ax=ax[0])

    A.plotModeNumber(
        R2_threshold=R2_threshold,
        mode_number_lim=mag_mode_number_lim,
        cmap=cmap_mag,
        tlim=tlim,
        flim=flim,
        xlabel=xlabel,
        ylabel='',
        title='magnetics',
        fontsize=fontsize,
        interpolation=interpolation,
        ax=ax[1])

    # Set colorbar label as desired
    cb0 = ax[0].images[0].colorbar
    cb0.set_label(r'$\mathregular{n}$', fontsize=fontsize)
    cb1 = ax[1].images[0].colorbar
    cb1.set_label(r'$\mathregular{n}$', fontsize=fontsize)

    # Annotate
    x0 = 2.165

    ax[0].text(x0, 22., 'n = 1', fontsize=fontsize)
    ax[0].text(x0, 42., 'n = 2', fontsize=fontsize)
    ax[0].text(x0, 62., 'n = 3', fontsize=fontsize)
    ax[0].text(x0, 82., 'n = 4', fontsize=fontsize)

    ax[1].text(x0, 22., 'n = 1', fontsize=fontsize)
    ax[1].text(x0, 42., 'n = 2', fontsize=fontsize)
    ax[1].text(x0, 62., 'n = 3', fontsize=fontsize)

    ax[1].text(2.1625, 2, '%i' % shot, fontsize=(fontsize - 3))

    plt.xlim(tlim)
    plt.ylim(flim)
    plt.tight_layout()

    plt.show()
