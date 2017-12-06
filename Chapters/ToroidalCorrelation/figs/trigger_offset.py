import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random_data as rd
import mitpci


# Shot parameters
shot = 167341
tlim = [1.0, 1.6]   # [tlim] = s

# Spectral-estimation parameters
Tens = 10e-3         # [Tens] = s
Nreal_per_ens = 20

# Unit conversions
Hz_per_kHz = 1e3

# Plotting parameters
gamma2xy_threshold = 0.25  # 0.5
all_positive = False
tlim = [1.15, 1.45]  # [tlim] = s
flim = [130., 180.]  # [flim] = kHz
figsize = (9, 5)
fontsize = 15
annotation_color = 'r'
annotation_linewidth = 4
annotation_linestyle = '-'


if __name__ == '__main__':
    # Load PCI interferometer data
    L = mitpci.interferometer.Lissajous(shot, tlim=tlim)
    Ph = mitpci.interferometer.Phase(L)

    # Compute toroidal mode-number spectrum
    # *without* compensation for trigger offset
    TorCorrUncomp = mitpci.interferometer.ToroidalCorrelation(
        Ph,
        trigger_offset=0.0,
        Tens=Tens,
        Nreal_per_ens=Nreal_per_ens)

    # Compute toroidal mode-number spectrum
    # *with* compensation for trigger offset
    TorCorrComp = mitpci.interferometer.ToroidalCorrelation(
        Ph,
        Tens=Tens,
        Nreal_per_ens=Nreal_per_ens)

    TorCorrUncomp.f /= Hz_per_kHz
    TorCorrUncomp.df /= Hz_per_kHz
    TorCorrComp.f /= Hz_per_kHz
    TorCorrComp.df /= Hz_per_kHz

    # Plot spectrum using "nearest" interpolation
    # to prevent aliasing in saved figured, as described
    # here:
    #
    #   https://github.com/matplotlib/matplotlib/issues/2972/
    #
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)

    TorCorrUncomp.plotModeNumber(
        gamma2xy_threshold=gamma2xy_threshold,
        all_positive=all_positive,
        tlim=tlim,
        flim=flim,
        xlabel=r'$\mathregular{t \; [s]}$',
        ylabel=r'$\mathregular{f \; [kHz]}$',
        fontsize=fontsize,
        interpolation='nearest',
        ax=ax[0])
    TorCorrComp.plotModeNumber(
        gamma2xy_threshold=gamma2xy_threshold,
        all_positive=all_positive,
        tlim=tlim,
        flim=flim,
        xlabel=r'$\mathregular{t \; [s]}$',
        ylabel='',
        fontsize=fontsize,
        interpolation='nearest',
        ax=ax[1])

    # Set colorbar label as desired
    cb0 = ax[0].images[0].colorbar
    cb0.set_label(r'$\mathregular{n}$', fontsize=fontsize)
    cb1 = ax[1].images[0].colorbar
    cb1.set_label(r'$\mathregular{n}$', fontsize=fontsize)

    # Annotate uncompensated spectrum
    e1 = Ellipse(
        (1.197, 136.),
        5. * TorCorrUncomp.dt,
        4. * TorCorrUncomp.df,
        fill=False,
        edgecolor=annotation_color,
        linewidth=annotation_linewidth,
        linestyle=annotation_linestyle)
    e2 = Ellipse(
        (1.315, 154.),
        5. * TorCorrUncomp.dt,
        4. * TorCorrUncomp.df,
        fill=False,
        edgecolor=annotation_color,
        linewidth=annotation_linewidth,
        linestyle=annotation_linestyle)
    ax[0].add_patch(e1)
    ax[0].add_patch(e2)

    ax[0].text(1.39, 131., '%i' % shot, fontsize=(fontsize - 2))
    ax[1].text(1.39, 131., '%i' % shot, fontsize=(fontsize - 2))

    ax[0].text(1.155, 176.5, '(a)', fontsize=fontsize)
    ax[1].text(1.155, 176.5, '(b)', fontsize=fontsize)

    plt.xlim(tlim)
    plt.ylim(flim)
    plt.tight_layout()

    plt.show()
