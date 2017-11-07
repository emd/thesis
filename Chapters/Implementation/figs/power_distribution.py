import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Design-parameter range
eta_R = np.linspace(0.0, 0.12, 101)  # Deflection efficiency of AOM
eta_P = np.linspace(0.1, 0.3, 151)    # Probe-beam splitter reflectivity
eta_R_design = 0.05
eta_P_design = 0.25

# Plotting parameters
figsize = (6, 5)
fontsize = 12
cmap = 'viridis'
cbar_orientation = 'vertical'

# Constants
P_source = 14e3  # [P_source] = mW
P_pci_0 = 300.   # [P_pci_0] = mW
eta_IL = 0.9     # Static optical throughput of AOM
eta_ZnSe = 0.17  # Reflectivity of phase-plate groove
wp = 2.7         # [wp] = mm, 1/e E radius of probe beam @ detector
wr = 4.3         # [wr] = mm, 1/e E radius of reference beam @ detector
s = 1.           # [s] = mm, side length of square detector element
D = 1.5          # [D] = mm, diameter of iris


def pci_power(eta_R, eta_P):
    return eta_IL * (1 - eta_R) * (1 - eta_P) * P_pci_0


def probe_beam_power(eta_R, eta_P):
    return (P_pci_0 - pci_power(eta_R, eta_P)) / (2 * eta_ZnSe)


def reference_beam_power(eta_R):
    return 0.5 * eta_IL * eta_R * P_source


def gaussian_integral_over_square(s, w):
    return (erf(s / (np.sqrt(2) * w))) ** 2


def gaussian_integral_over_circle(D, w):
    return 1 - np.exp(-(D ** 2) / (2 * (w ** 2)))


if __name__ == '__main__':
    # Construct 2d computational grid
    RR, PP = np.meshgrid(eta_R, eta_P)

    # Compute total beam powers
    Ppci = pci_power(RR, PP)
    Pp = probe_beam_power(RR, PP)
    Pr = reference_beam_power(RR)

    # Compute beam powers passing through iris
    Pp_iris = Pp * gaussian_integral_over_circle(D, wp)
    Pr_iris = Pr * gaussian_integral_over_circle(D, wr)

    # Compute on-axis intensities, which are also reasonable proxies for
    # the average intensities over the face of the interferometer detector
    # because the 1/e E radii are "much larger" than the element size
    Ip = 2 * Pp / (np.pi * (wp ** 2))
    Ir = 2 * Pr / (np.pi * (wr ** 2))
    Idc = Ip + Ir
    Iac = 2 * np.sqrt(Ip * Ir)
    Tmult = 2 * Iac / (Iac + Idc)  # Transfer-function multiple

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    levels00 = np.arange(0.75, 1.01, 0.025)
    C00 = axes[0, 0].contourf(RR, PP, Tmult, levels00, cmap=cmap)
    cb00 = plt.colorbar(C00, ax=axes[0, 0], orientation=cbar_orientation)
    cb00.set_ticks(levels00[::2])
    axes[0, 0].set_ylabel(
        r'$\eta_P$',
        fontsize=fontsize)
    axes[0, 0].set_title(
        r'$2 \, I_{\mathrm{AC}} \, / \, (I_{\mathrm{DC}} + I_{\mathrm{AC}})$',
        fontsize=fontsize)

    levels01 = np.arange(20, 121, 10)
    C01 = axes[0, 1].contourf(RR, PP, Idc + Iac, levels01, cmap=cmap)
    cb01 = plt.colorbar(C01, ax=axes[0, 1], orientation=cbar_orientation)
    cb01.set_ticks(levels01[::2])
    axes[0, 1].set_title(
        r'$I_{\mathrm{DC}} + I_{\mathrm{AC}} \; [\mathrm{mW} / \mathrm{mm}^2]$',
        fontsize=fontsize)

    levels10 = np.arange(160, 251, 10)
    C10 = axes[1, 0].contourf(RR, PP, Ppci, levels10, cmap=cmap)
    cb10 = plt.colorbar(C10, ax=axes[1, 0], orientation=cbar_orientation)
    cb10.set_ticks(levels10[::2])
    axes[1, 0].set_xlabel(
        r'$\eta_R$',
        fontsize=fontsize)
    axes[1, 0].set_ylabel(
        r'$\eta_P$',
        fontsize=fontsize)
    axes[1, 0].set_title(
        r'$\mathcal{P}_{\mathrm{pci}} \; [\mathrm{mW}]$',
        fontsize=fontsize)

    levels11 = np.arange(25, 76, 5)
    C11 = axes[1, 1].contourf(RR, PP, Pp_iris + Pr_iris, levels11, cmap=cmap)
    cb11 = plt.colorbar(C11, ax=axes[1, 1], orientation=cbar_orientation)
    cb11.set_ticks(levels11[::2])
    axes[1, 1].set_xlabel(
        r'$\eta_R$',
        fontsize=fontsize)
    axes[1, 1].set_title(
        r'$Q_{\mathrm{heat}} \; [\mathrm{mW}]$',
        fontsize=fontsize)

    for ax in axes.flatten():
        ax.hlines(
            eta_P_design, eta_R[0], eta_R[-1],
            linestyle='--', colors='k')
        ax.plot(
            eta_R_design, eta_P_design,
            'D', c='darkred')

    plt.tight_layout()

    plt.show()
