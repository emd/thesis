import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct
from gadata import gadata


shots = [171536, 171538]
rho = [0.5, 0.8]

stationary_windows = [
    [2.5, 3.2],
    [2.0, 2.8]
]
profile_times = [2.75, 2.20]

# Plotting parameters
# figsize = (9, 9)
figsize = (8, 8)
cols = get_distinct(len(shots))
linewidth = 2
fontsize = 15
rotation = 90
alpha = 0.25
profile_linestyle = '--'
tlim = [1.0, 3.5]


class PointName(object):
    def __init__(self, point_name, label, norm=1, Nsmooth=None):
        self.point_name = point_name
        self.norm = norm
        self.label = label
        self.Nsmooth = Nsmooth

    def box_smooth(self, x):
        'Box smooth `x` by `self.Nsmooth`; note that there are edge effects'
        if self.Nsmooth is None:
            return x
        else:
            box = np.ones(self.Nsmooth) / self.Nsmooth
            return np.convolve(x, box, mode='same')


Ip_label = (
    r'$\mathregular{I_p}$'
    #+ '\n'
    + r'$\;$'
    + r'$\mathregular{[MA]}$')
H_THH98Y2_label = r'$\mathregular{H_{98,y2}}$'
betan_label = r'$\mathregular{\beta_N}$'
pinj_label = (
    r'$\mathregular{P_{inj}}$'
    #+ '\n'
    + r'$\;$'
    + r'$\mathregular{[MW]}$')
tinj_label = (
    r'$\mathregular{T_{inj}}$'
    #+ '\n'
    + r'$\;$'
    + r'$\mathregular{[N \cdot m]}$')
echpwrc_label = (
    r'$\mathregular{P_{EC}}$'
    # + '\n'
    + r'$\;$'
    + r'$\mathregular{[MW]}$')
density_label = (
    r'$\mathregular{\overline{n}_e}$'
    # + '\n'
    + r'$\;$'
    + r'$\mathregular{[10^{19} \, m^{-3}]}$')
fs04_label = (
    r'$\mathregular{D_{\alpha}}$'
    # + '\n'
    + r'$\;$'
    + r'$\mathregular{[a.u.]}$')

point_names = [
    PointName('Ip', Ip_label, norm=1e6, Nsmooth=10),
    PointName('pinj', pinj_label, norm=1e3, Nsmooth=1000),
    PointName('tinj', tinj_label, norm=1, Nsmooth=1000),
    PointName('echpwrc', echpwrc_label, norm=1e6, Nsmooth=100),
    PointName('density', density_label, norm=1e13, Nsmooth=100),
    PointName('betan', betan_label, norm=1),
    PointName('H_THH98Y2', H_THH98Y2_label, norm=1),
    PointName('fs04', fs04_label, norm=1e16, Nsmooth=10),
]

# List for ylims of each axis; for some reason, ax.get_ylim()
# only gets the ylims of the most recently plotted trace but
# does *not* reflect the true y-axis extent...
#
# Use this workaround for now... (wouldn't be necessary if
# GA would actually FUCKING update matplotlib...)
ylims = [
    1.5,
    5.,
    2.,
    5.,
    6.,
    2.5,
    1.5,
    10.
]


if __name__ == '__main__':
    Nrows = 4
    Ncolumns = 2
    fig, ax = plt.subplots(
        Nrows,
        Ncolumns,
        figsize=figsize,
        sharex=True)

    for sind, shot in enumerate(shots):
        for pind, point_name in enumerate(point_names):
            # Load data
            data = gadata(point_name.point_name, shot)

            # Convert from ms to s
            t = data.xdata * 1e-3

            # Smooth trace and apply normalization
            x = point_name.box_smooth(data.zdata)
            x /= point_name.norm

            # Find points within `tlim`
            tind = np.where(np.logical_and(
                t >= tlim[0],
                t <= tlim[1]))[0]

            # Plot trace
            rowind = pind % Nrows
            colind = pind // Nrows
            ax[rowind, colind].plot(
                t[tind],
                x[tind],
                color=cols[sind],
                linewidth=linewidth)
            ax[rowind, colind].set_ylabel(
                point_name.label,
                rotation=rotation,
                fontsize=fontsize)

            # Annotate trace w/ stationary window used for
            # spectral analysis and time slice corresponding
            # to modeling work
            ax[rowind, colind].fill_betweenx(
                [0, ylims[pind]],
                stationary_windows[sind][0],
                x2=stationary_windows[sind][1],
                color=cols[sind],
                alpha=alpha)
            ax[rowind, colind].vlines(
                profile_times[sind],
                0,
                ylims[pind],
                color=cols[sind],
                linestyle=profile_linestyle,
                linewidth=linewidth)

            if sind == 1:
                ax[rowind, colind].set_ylim([0, ylims[pind]])

    for column in np.arange(Ncolumns):
        ax[-1, column].set_xlabel(
            r'$\mathregular{t \; [s]}$',
            fontsize=fontsize)

    for sind, shot in enumerate(shots):
        x0 = 1.075
        y0 = 0.3
        dy = 0.2
        ax[0, 0].annotate(
            '%i' % shot,
            (x0, y0 - (sind * dy)),
            color=cols[sind],
            fontsize=(fontsize - 2))

    plt.xlim(tlim)
    plt.tight_layout()
    plt.show()
