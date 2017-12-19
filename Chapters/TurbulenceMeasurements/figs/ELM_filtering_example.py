import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct

import mitpci
import random_data as rd
from gadata import gadata


shot = 171536
shot = 171538
# tlim_data = [2.9, 3.0]
tlim_data = [2.1, 2.3]

# Divertor D-alpha parameters
Dalpha_pointname = 'fs04'
Dalpha_norm = 1e16

# ELM-filtering parameters used for S(f), S(k,f), & S(k) analysis
sigma_mult = 3
debounce_dt = 0.5e-3
window_fraction = [0.2, 0.8]

# Plotting parameters
figsize = (8, 5)
# tlim = [2.92, 2.98]
tlim = [2.13, 2.23]
trace_color = get_distinct(1)[0]
linewidth = 2
fontsize = 15
spike_color = 'lightgray'


if __name__ == '__main__':
    # Load interferometer data & compute phase
    L = mitpci.interferometer.Lissajous(shot, tlim=tlim_data)
    Ph_int = mitpci.interferometer.Phase(L)

    # Load divertor D-alpha data and convert to appropriate units
    Dalpha = gadata(Dalpha_pointname, shot)
    Dalpha.xdata /= 1e3             # convert from ms -> s
    Dalpha.zdata /= Dalpha_norm
    tind = np.where(np.logical_and(
        Dalpha.xdata >= tlim[0],
        Dalpha.xdata <= tlim[1]))[0]

    # Look for ELMs in interferometer data
    SH = rd.signals.SpikeHandler(
        Ph_int.x, Fs=Ph_int.Fs, t0=Ph_int.t0,
        sigma_mult=sigma_mult, debounce_dt=debounce_dt)

    # Plot trace with identified ELMs along w/ D-alpha data
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize)

    SH.plotTraceWithSpikeColor(
        Ph_int.x,
        Ph_int.t(),
        window_fraction=window_fraction,
        ax=ax[0],
        spike_color=spike_color)
    SH.plotTraceWithSpikeColor(
        Dalpha.zdata[tind],
        Dalpha.xdata[tind],
        window_fraction=window_fraction,
        ax=ax[1],
        spike_color=spike_color)

    for i in np.arange(2):
        L = ax[i].lines[0]
        L.set_color(trace_color)
        L.set_linewidth(linewidth)

    ax[0].set_xlim(tlim)
    ax[0].set_ylim([-0.06, 0.06])
    ax[1].set_ylim([0, 10])

    ax[0].set_xlabel('')
    ax[0].set_ylabel(
        r'$\mathregular{\widetilde{\phi} \; [rad]}$',
        fontsize=fontsize)
    ax[1].set_xlabel(
        r'$\mathregular{t \; [s]}$',
        fontsize=fontsize)
    ax[1].set_ylabel(
        r'$\mathregular{D_{\alpha} \; [a.u.]}$',
        fontsize=fontsize)

    ax[1].annotate(
        '%i' % shot,
        (2.219, 9),
        fontsize=(fontsize - 2))

    plt.tight_layout()
    plt.show()
