import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle

import random_data as rd
import mitpci
import filters


# Spectral-estimation parameters
p = 6
Nk = 1000

# Plotting parameters
figsize = (10, 10.5)
fontsize = 18
mpl.rcParams['xtick.labelsize'] = fontsize - 3
mpl.rcParams['ytick.labelsize'] = fontsize - 3
cmap = 'viridis'
cborientation = 'horizontal'
flim = [10, 1600]           # [flim] = kHz
flim_zoom = [500, 1600]     # [flim_zoom] = kHz
vlim_Skf = [3e-10, 1e-4]
vlim_Skf_zoom = [3e-10, 5e-8]
annotation_linestyle = '--'
annotation_linewidth = 0.5
plot_labels = np.array([['a', 'b'], ['c', 'd']])


if __name__ == '__main__':
    # Load previously computed correlation functions
    fname = './corrs.pkl'
    with open(fname, 'rb') as f:
        corr_list = pickle.load(f)
        rho_list = pickle.load(f)
        shot_list = pickle.load(f)
        tlim_list = pickle.load(f)
        tau_list = pickle.load(f)
        Nreal_list = pickle.load(f)

    Nwin = len(corr_list)

    fig, axs = plt.subplots(
        2, 2,
        figsize=figsize)

    for ind, w in enumerate(np.arange(Nwin)):
        print 'Processing window %i of %i' % (w, Nwin - 1)

        corr = corr_list[w]
        rho = rho_list[w]
        shot = shot_list[w]
        tlim = tlim_list[w]
        tau = tau_list[w]
        Nreal = Nreal_list[w]

        # Compute 2d autospectral density and plot
        asd2d = mitpci.pci.TwoDimensionalAutoSpectralDensity(
            corr,
            burg_params={'p': p, 'Nk': Nk})

        xlabel = r'$\mathregular{k_R \; [cm^{-1}]}$'

        # if ind == 0:
        #     ylabel = r'$\mathregular{f \; [kHz]}$'
        # else:
        #     ylabel = ''
        ylabel = r'$\mathregular{f \; [kHz]}$'

        cbsymbol = r'$\mathregular{S_{\phi,\phi}(k,f)}$'
        cbunits = r'$\mathregular{[rad^2 / \, (kHz \cdot cm^{-1})]}$'
        cblabel = cbsymbol + ' ' + cbunits

        asd2d.plotSpectralDensity(
            flim=flim,
            vlim=vlim_Skf,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            cblabel=cblabel,
            cborientation=cborientation,
            fontsize=fontsize,
            ax=axs[0, ind])

        asd2d.plotSpectralDensity(
            flim=flim_zoom,
            vlim=vlim_Skf_zoom,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            cblabel=cblabel,
            cborientation=cborientation,
            fontsize=fontsize,
            ax=axs[1, ind])

        # Add ECH heating location
        x0 = -24.5
        axs[0, ind].annotate(
            r'$\mathregular{\rho_{ECH} = %.1f}$' % rho,
            (x0, 1475),
            fontsize=(fontsize - 1),
            color='white')
        axs[1, ind].annotate(
            r'$\mathregular{\rho_{ECH} = %.1f}$' % rho,
            (x0, 1510),
            fontsize=(fontsize - 1),
            color='white')

        # Add shot numbers and time windows
        axs[0, ind].annotate(
            '%i, [%.2f, %.2f] s' % (shot, tlim[0], tlim[1]),
            (x0, 1360),
            fontsize=(fontsize - 6),
            color='white')
        axs[1, ind].annotate(
            '%i, [%.2f, %.2f] s' % (shot, tlim[0], tlim[1]),
            (x0, 1430),
            fontsize=(fontsize - 6),
            color='white')

        # Add plot label
        axs[0, ind].annotate(
            r'(%s)' % plot_labels[0, ind],
            (x0, 50),
            fontsize=(fontsize - 1),
            color='white')
        axs[1, ind].annotate(
            r'(%s)' % plot_labels[1, ind],
            (x0, 535),
            fontsize=(fontsize - 1),
            color='white')


    # Add branch annotations:
    # -----------------------
    # Points in each branch
    pt_536 = (10., 900.)
    pt_538 = (6.8, 700.)

    # Assume that each branch passes through origin
    src_536 = (0, 0)
    src_538 = (0, 0)

    # Slopes...
    m_536 = pt_536[1] / pt_536[0]
    m_538 = pt_538[1] / pt_538[0]

    # Destination points
    dst_536 = (flim[1] / m_536, flim[1])
    dst_538 = (flim[1] / m_538, flim[1])

    # Overplot branch boundaries on S(k,f):
    # -------------------------------------
    # S(k, f) => k in row space, f in column space
    #
    # NOTE: Specifying `lwr` keyword means that `coord_lines`
    # will be uniformly spaced in k and that averaging will
    # be done over a narrow range in k at each frequency.
    # To get S(k) we technically want the reverse -- averaging
    # over a narrow range in f at each wavenumber. However,
    # because the branch "bends", particularly near the low-k
    # cutoff of the PCI, this technically correct method
    # introduces artifacts into the S(k)... Because we're
    # examining a narrow slice in S(k,f), it is OK to use
    # the former method of averaging over narrow range in k
    # at each frequency, especially as it allows us to avoid
    # the "bend" biasing.
    lpc_kwargs = {'N': 100, 'lwr': 2.5, 'L': 11}

    coord_lines_536 = rd.utilities.line_profile_coordinates(
        src_536, dst_536, **lpc_kwargs)
    coord_lines_538 = rd.utilities.line_profile_coordinates(
        src_538, dst_538, **lpc_kwargs)

    axs[0, 0].plot(
        coord_lines_536[0, :, 0],
        coord_lines_536[1, :, 0],
        c='white',
        linestyle=annotation_linestyle,
        linewidth=annotation_linewidth)
    axs[0, 0].plot(
        coord_lines_536[0, :, -1],
        coord_lines_536[1, :, -1],
        c='white',
        linestyle=annotation_linestyle,
        linewidth=annotation_linewidth)

    axs[0, 1].plot(
        coord_lines_538[0, :, 0],
        coord_lines_538[1, :, 0],
        c='white',
        linestyle=annotation_linestyle,
        linewidth=annotation_linewidth)
    axs[0, 1].plot(
        coord_lines_538[0, :, -1],
        coord_lines_538[1, :, -1],
        c='white',
        linestyle=annotation_linestyle,
        linewidth=annotation_linewidth)


    for rind in np.arange(2):
        for cind in np.arange(2):
            axs[rind, cind].set_xlim(
                asd2d.k[0] / 1e2,
                asd2d.k[-1] / 1e2)

            if rind == 0:
                axs[rind, cind].set_ylim(flim)
            else:
                axs[rind, cind].set_ylim(flim_zoom)

    plt.tight_layout()
    plt.show()
