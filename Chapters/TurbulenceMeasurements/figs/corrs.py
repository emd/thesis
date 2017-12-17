import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle

import random_data as rd
import mitpci
import filters

# Windows for analysis: includes shot, time window, & ECH location
from windows import windows


# PCI channel region
channel_region = 'all'  # may be in {'all', 'board7', 'board8', 'central'}

# High-pass temporal filter
hpf = mitpci.interferometer.demodulated._hpf

# ELM-filtering parameters
sigma_mult = 3
debounce_dt = 0.5e-3  # [debounce_dt] = s; less than ELM spacing
window_fraction = [0.2, 0.8]

# Spectral-estimation parameters
Treal = 0.5e-3      # [Treal] = s; less than ELM spacing
maxinterp = 4

# Plotting parameters
flim = [10, 2000]  # [flim] = kHz


if __name__ == '__main__':
    Nwin = len(windows)

    # Determine which PCI channels to load
    if str.lower(channel_region) == 'all':
        digitizer_channels = 1 + np.arange(16)
    elif str.lower(channel_region) == 'board7':
        digitizer_channels = 1 + np.arange(8)
    elif str.lower(channel_region) == 'board8':
        digitizer_channels = 9 + np.arange(8)
    elif str.lower(channel_region) == 'central':
        digitizer_channels = np.array([6, 7, 8, 9, 12, 13])

    print '\n%s channels: %s' % (channel_region, digitizer_channels)

    # Lists to accumulate correlation functions
    corr_list = []
    rho_list = []
    shot_list = []
    tlim_list = []
    tau_list = []
    Nreal_list = []

    for window_index, w in enumerate(windows):
        print '\nWindow %i of %i' % (window_index + 1, Nwin)

        rho = w.rho_ECH
        shot = w.shot
        tlim = w.tlim
        tau = w.tau

        # Load PCI data
        Ph_pci = mitpci.pci.Phase(
            shot, digitizer_channels, tlim=tlim, tau=tau)

        # Load interferometer data, from which ELM timing is determined
        L = mitpci.interferometer.Lissajous(shot, tlim=tlim)
        Ph_int = mitpci.interferometer.Phase(L, filt=hpf)

        # Locate ELMs
        SH = rd.signals.SpikeHandler(
            Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
            sigma_mult=sigma_mult, debounce_dt=debounce_dt)

        # Initialize `ComplexCorrelationFunction` instance,
        # noting that ensemble averaging will be performed
        # *after* ELM filtering (explaining `Nreal_per_ens = 1`)
        corr = mitpci.pci.ComplexCorrelationFunction(
            Ph_pci,
            tlim=[Ph_pci.t0, Ph_pci.t0 + Treal],
            Nreal_per_ens=1)

        # Zero the spectral estimate, as we do not yet know
        # if the initial ensemble is free of ELMs
        corr.Gxy *= 0
        Nreal = 0

        ELM_free_tstart, ELM_free_tstop = SH._getSpikeFreeTimeWindows(
            window_fraction=window_fraction)

        # The time series will be split into ELM-free blocks.
        # These blocks will then be subdivided into realizations.
        print ''
        Nblock = len(ELM_free_tstart[:-1]) - 1

        for block, t0_block in enumerate(ELM_free_tstart[:-1]):
            print 'Computing correlation for block %i of %i' % (block, Nblock)

            # Find first realization in `block`
            realization = 0
            t0_real = t0_block + (realization * corr.dt)
            tf_real = t0_real + corr.dt

            # Compute correlations for realizations that lie
            # fully within the ELM-free block
            while tf_real <= ELM_free_tstop[block]:
                corr_tmp = mitpci.pci.ComplexCorrelationFunction(
                    Ph_pci,
                    tlim=[t0_real, tf_real],
                    Nreal_per_ens=1,
                    Npts_per_real=corr.Npts_per_real,
                    print_status=False)

                # Accumulate
                corr.Gxy += corr_tmp.Gxy
                Nreal += 1

                # Find next realization in `block`
                realization += 1
                t0_real = t0_block + (realization * corr.dt)
                tf_real = t0_real + corr.dt

        # Compute ensemble-averaged correlation function and plot
        corr.Gxy /= Nreal
        corr.interpolate(maxinterp)
        corr.plotNormalizedCorrelationFunction(flim=flim)
        fname = ('./corr_rho%.1f_%i_%.1f_to_%.1f_%s_Nreal%i.png'
                 % (rho, shot, tlim[0], tlim[1], channel_region, Nreal))
        plt.savefig(fname)
        plt.close()

        # Append correlation function and corresponding parameters
        corr_list.append(corr)
        rho_list.append(rho)
        shot_list.append(shot)
        tlim_list.append(tlim)
        tau_list.append(tau)
        Nreal_list.append(Nreal)

    # Save correlation functions
    fname = './corrs.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(corr_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(rho_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(shot_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tlim_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tau_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(Nreal_list, f, pickle.HIGHEST_PROTOCOL)
