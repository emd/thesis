import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random_data as rd
import mitpci


# Shot parameters
shot = 167341
tlim = [0.9, 1.5]   # [tlim] = s

# Spectral-estimation parameters
Tens = 10e-3         # [Tens] = s
Nreal_per_ens = 10

# Unit conversions
Hz_per_kHz = 1e3

# Plotting parameters
gamma2xy_threshold = 0.5
all_positive = False
flim = [80., 180.]  # [flim] = kHz
figsize = (7, 5)
fontsize = 15
annotation_color = 'r'
annotation_linewidth = 3
annotation_linestyle = '-'


if __name__ == '__main__':
    # Load PCI interferometer data
    L = mitpci.interferometer.Lissajous(shot, tlim=tlim)
    Ph = mitpci.interferometer.Phase(L)

    # Compute toroidal mode-number spectrum
    TorCorr = mitpci.interferometer.ToroidalCorrelation(
        Ph,
        Tens=Tens,
        Nreal_per_ens=Nreal_per_ens)

    TorCorr.f /= Hz_per_kHz
    TorCorr.df /= Hz_per_kHz

    # Plot spectrum using "nearest" interpolation
    # to prevent aliasing in saved figured, as described
    # here:
    #
    #   https://github.com/matplotlib/matplotlib/issues/2972/
    #
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    TorCorr.plotModeNumber(
        gamma2xy_threshold=gamma2xy_threshold,
        all_positive=all_positive,
        tlim=tlim,
        flim=flim,
        xlabel=r'$\mathregular{t \; [s]}$',
        ylabel=r'$\mathregular{f \; [kHz]}$',
        fontsize=fontsize,
        interpolation='nearest',
        ax=ax)

    # Set colorbar label as desired
    cb0 = ax.images[0].colorbar
    cb0.set_label(r'$\mathregular{n}$', fontsize=fontsize)
    # cb1 = ax[1].images[0].colorbar
    # cb1.set_label(r'$\mathregular{n}$', fontsize=fontsize)

    # Annotate uncompensated spectrum
    xmult = 10.
    ymult = 25.
    e1 = Ellipse(
        (1.147, 146.),
        xmult * TorCorr.dt,
        ymult * TorCorr.df,
        fill=False,
        edgecolor=annotation_color,
        linewidth=annotation_linewidth,
        linestyle=annotation_linestyle)
    e2 = Ellipse(
        (1.067, 106.),
        xmult * TorCorr.dt,
        ymult * TorCorr.df,
        fill=False,
        edgecolor=annotation_color,
        linewidth=annotation_linewidth,
        linestyle=annotation_linestyle)
    ax.add_patch(e1)
    ax.add_patch(e2)

    ax.text(1.425, 173., '%i' % shot, fontsize=(fontsize - 2))

    plt.xlim(tlim)
    plt.ylim(flim)
    plt.tight_layout()

    plt.show()
