import numpy as np
import matplotlib.pyplot as plt


# Plotting parameters
kmax = 3    # [kmax] = k_{fsv}
dk = 0.01   # [dk] = [kmax]
linewidth = 2
fontsize = 12


if __name__ == '__main__':
    k = np.arange(-kmax, kmax + dk, dk)

    plt.figure()

    plt.plot(k, 0.25 * np.sinc(k),
             # label='FSV included',
             label='finite sampling volume',
             c='C0', linestyle='-', linewidth=linewidth)
    plt.plot(k, 0.25 * np.ones(len(k)),
             # label='no FSV',
             label='no finite sampling volume',
             c='C0', linestyle='--', linewidth=linewidth)
    plt.axhline(0, c='k')

    plt.xlim([-kmax, kmax])
    plt.ylim([-0.1, 0.35])
    plt.xlabel('$k \; [k_{\mathrm{fsv}}]$', fontsize=fontsize)
    plt.ylabel('$T_{\mathrm{het}}(k)$ [unitless]', fontsize=fontsize)
    # plt.legend(loc='best')
    plt.legend(loc='upper right')

    plt.show()
