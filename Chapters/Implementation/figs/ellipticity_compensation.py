import numpy as np
import matplotlib.pyplot as plt
from distinct_colours import get_distinct
import random_data as rd
import mitpci


shot = 170864
tlim = [0.75, 0.80]
sl = slice(None, None, 100)

# Plotting parameters
fontsize = 14
linewidth = 2
cols = get_distinct(2)


if __name__ == '__main__':
    Lraw = mitpci.interferometer.Lissajous(
        shot, tlim=tlim, fit=False, compensate=False)
    L = mitpci.interferometer.Lissajous(
        shot, tlim=tlim)

    plt.figure()

    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.plot(
        Lraw.I.x[sl],
        Lraw.Q.x[sl],
        linewidth=linewidth,
        c=cols[0],
        label='raw')
    plt.plot(
        L.I.x[sl],
        L.Q.x[sl],
        linewidth=linewidth,
        c=cols[1],
        label='compensated')

    plt.gca().set_aspect('equal')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('$\mathregular{I \; [V]}$', fontsize=fontsize)
    plt.ylabel('$\mathregular{Q \; [V]}$', fontsize=fontsize)
    plt.legend(ncol=2, loc='lower right')
    plt.text(
        -3.9, 3.6,
        '%i, [%.2f, %.2f] s' % (shot, tlim[0], tlim[1]),
        fontsize=(fontsize - 2))

    plt.show()
