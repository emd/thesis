import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct


# Plotting parameters
cols = get_distinct(1)
linewidth = 2
alpha = 0.5
fontsize = 15


class Measurements(object):
    def __init__(self, var, shot):
        skip_header = 6

        if var in set(('ti1', 'ti', 'trot', 'imp')):
            usecols = (0, 1, 3, 6)

            if var == 'ti1':
                var = var[:-1]

            if var == 'trot':
                skip_footer = 3
            else:
                skip_footer = 0

        elif var == 'ne':
            usecols = (0, 1, 3, 5)
            skip_footer = 0
        elif var == 'te':
            usecols = (0, 1, 3, 5)
            skip_header = 5
            skip_footer = 0

        self.var = var

        # Parse input to locate corresponding file
        if shot == 171536:
            dirtime = 2750
            gaproftime = 2753
        elif shot == 171538:
            dirtime = 2200
            gaproftime = 2202
        else:
            raise ValueError('Unrecognized shot!')

        dname = '%i/%ibis/uncertainties/gaprofiles' % (shot, dirtime)
        fname = '%s/d%s.rho%i.0%i' % (dname, self.var, shot, gaproftime)

        if var == 'imp':
            fname += '_Carbon'

        # Load data
        d = np.genfromtxt(
            fname,
            skip_header=skip_header,
            skip_footer=skip_footer,
            usecols=usecols)

        # Parse data
        self.rho = d[:, 0]      # radial coordinate
        self.y = d[:, 1]        # profile
        self.yerr = d[:, 2]     # uncertainty in profile
        self.valid = d[:, 3]    # point used in fit?

        # Determine viewing geometry
        if self.var in set(('ti', 'trot', 'imp')):
            self.view = np.genfromtxt(
                fname,
                dtype='string',
                skip_header=skip_header,
                skip_footer=skip_footer,
                usecols=(7))
        elif var == 'ne':
            self.view = np.genfromtxt(
                fname,
                dtype='string',
                skip_header=skip_header,
                skip_footer=skip_footer,
                usecols=(6))
        else:
            # For some reason, viewing geometry *not* in te file... WTF.
            # So get data from ne file
            fname = '%s/d%s.rho%i.0%i' % (dname, 'ne', shot, gaproftime)
            self.view = np.genfromtxt(
                fname,
                dtype='string',
                skip_header=6,  # different than the te value... again, WTF
                usecols=(6))

    def plot(self, rho_lim=[0, 1], drho=0.,
             valid=True, exclude_view=None,
             ax=None, color=cols[0], marker='o', markersize=5):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if valid and (exclude_view is not None):
            vind = np.where(np.logical_and(
                self.valid == 1,
                self.view != exclude_view))[0]
        elif valid:
            vind = np.where(self.valid == 1)[0]
        elif (exclude_view is not None):
            vind = np.where(self.view != exclude_view)[0]
        else:
            vind = slice(None, None)

        rhoind = np.where(np.logical_and(
            self.rho[vind] >= rho_lim[0],
            self.rho[vind] <= rho_lim[1]))[0]

        ax.errorbar(
            self.rho[vind][rhoind] + drho,
            self.y[vind][rhoind],
            yerr=self.yerr[vind][rhoind],
            color=color,
            fmt=marker,
            markersize=markersize)

        plt.show()

        return ax


class Uncertainty(object):
    def __init__(self, var, shot, rho=None, skiprows=5):
        if var == 'ni1' or var == 'ti1':
            var = var[:-1]

        self.var = var

        # Parse input to locate corresponding file
        if shot == 171536:
            dirtime = 2750
            gaproftime = 2753
        elif shot == 171538:
            dirtime = 2200
            gaproftime = 2202
        else:
            raise ValueError('Unrecognized shot!')

        dname = '%i/%ibis/uncertainties/gaprofiles' % (shot, dirtime)
        fname = '%s/d%s.xy.%i.0%i' % (dname, self.var, shot, gaproftime)

        if var == 'imp':
            fname += '_Carbon'

        # Load data
        d = np.loadtxt(fname, skiprows=skiprows)

        # Parse data
        self.rho = d[:, 0]      # radial coordinate
        self.y = d[:, 1]        # profile
        self.yerr = d[:, 2]     # uncertainty in profile
        self.yperr = d[:, 3]    # uncertainty in first derivative

        # Determine relative error in y
        self.y_relerr = self.yerr / self.y

        # Determine a / Ly
        yp = np.gradient(self.y) / np.gradient(self.rho)
        self.aLy = -yp / self.y

        # Determine relative error in a / Ly
        term1 = (self.yerr / self.y) ** 2
        term2 = (self.yperr / yp) ** 2
        self.aLy_relerr = np.sqrt(term1 + term2)

        # If requested, interpolate data onto new radial grid
        if rho is not None:
            self.y = np.interp(rho, self.rho, self.y)
            self.yerr = np.interp(rho, self.rho, self.yerr)
            self.yperr = np.interp(rho, self.rho, self.yperr)
            self.y_relerr = np.interp(rho, self.rho, self.y_relerr)
            self.aLy = np.interp(rho, self.rho, self.aLy)
            self.aLy_relerr = np.interp(rho, self.rho, self.aLy_relerr)
            self.rho = rho

    def plot(self, rholim=[0, 1]):
        rhoind = np.where(np.logical_and(
            self.rho >= rholim[0],
            self.rho < rholim[1]))[0]

        fig, axs = plt.subplots(1, 2, sharex=True)

        axs[0].plot(
            self.rho[rhoind],
            self.y_relerr[rhoind],
            color=cols[0],
            linewidth=linewidth)

        axs[1].semilogy(
            self.rho[rhoind],
            self.aLy_relerr[rhoind],
            color=cols[0],
            linewidth=linewidth)

        axs[0].set_xlabel(
            r'$\mathregular{\rho}$',
            fontsize=fontsize)
        axs[0].set_ylabel(
            r'$\mathregular{\sigma_{%s} / %s}$' % (self.var, self.var),
            fontsize=fontsize)

        axs[1].set_xlabel(
            r'$\mathregular{\rho}$',
            fontsize=fontsize)
        axs[1].set_ylabel(
            r'$\mathregular{\sigma_{L_{%s}} / L_{%s}}$' % (self.var, self.var),
            fontsize=fontsize)

        plt.tight_layout()
        plt.show()

        return


if __name__ == '__main__':
    shot = 171536
    drho = 0.005
    rho = np.arange(0, 1 + drho, drho)

    for profile in ['ne', 'te', 'ti', 'trot']:
        u = Uncertainty(profile, shot, rho=rho)
        u.plot()
