import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from distinct_colours import get_distinct


# Shots analyzed
shots = np.array([
    171536,
    171538
])

# Corresponding analysis times
# [times] = ms
times = np.array([
    2750,
    2200
])

# Plotting parameters
rholim = [0.0, 0.95]
gammaE_lim = (-10, 50)
Nsmooth = 5
figsize = (9, 3.5)
linewidth = 2
fontsize = 15
cols = get_distinct(len(shots))


def get_Er_force_balance(shot):
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
    skiprows = 5

    # Load density data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'ne', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    rho = d[:, 0]   # radial coordinate
    ne = d[:, 1]    # [ne] = 10^{19} m^{-3}

    # Load temperature data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'te', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    te = d[:, 1]    # [te] = keV

    # Load toroidal rotation data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'trot', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    trot = d[:, 1]  # [trot] = rad / s

    # Load poloidal-field data
    dname = '../doppler_shift/%i/%ibis/geometry' % (shot, dirtime)

    fname = '%s/RHORZ.pkl' % dname
    with open(fname, 'rb') as f:
        rho_geo = pickle.load(f)

    fname = '%s/Bp.pkl' % dname
    with open(fname, 'rb') as f:
        Bp_geo = pickle.load(f)

    # Find Bp along outboard midplane and interpolate
    # onto desired computational grid
    ind = np.where(rho_geo == np.min(rho_geo))
    Bp = np.interp(
        rho,
        np.squeeze(rho_geo[ind[0], ind[1]:]),
        np.squeeze(Bp_geo[ind[0], ind[1]:]))

    # Unit conversions
    R0 = 1.78
    a = 0.56
    R = R0 + (a * rho)
    vtor = trot * R     # [vtor] = m / s
    ne *= 1e19          # [ne] = m^{-3}
    te *= 1e3           # [te] = eV

    # Electron pressure gradient. Note that
    #
    #       [dpdr] = m^{-3} * eV / m
    #
    # such that division by electron charge `e`
    # will naturally cancel the "e" in `eV`
    dpdr = np.gradient(ne * te) / np.gradient(R)

    # Approximate ion diamagnetic term as
    # half of the negative electron diamagnetic term...
    # Seems to give good agreement with full field,
    # but not particularly happy about how jenky this is...
    Zeff = 2
    diamag_term = dpdr / ne / Zeff
    vtor_term = vtor * Bp
    vpol_term = 0  # neglecting this term...

    Er = diamag_term + vtor_term - vpol_term

    return rho, Er


def get_Er_uncertainty_force_balance(shot, rhointerp=None):
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
    skiprows = 5

    # Load density data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'ne', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    rho = d[:, 0]   # radial coordinate
    ne = d[:, 1]    # [ne] = 10^{19} m^{-3}
    ne_err = d[:, 2]
    nep_err = d[:, 3]

    # Load temperature data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'te', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    te = d[:, 1]    # [te] = keV
    te_err = d[:, 2]
    tep_err = d[:, 3]

    # Load toroidal rotation data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'trot', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    trot = d[:, 1]  # [trot] = rad / s
    trot_err = d[:, 2]

    # Load poloidal-field data
    dname = '../doppler_shift/%i/%ibis/geometry' % (shot, dirtime)

    fname = '%s/RHORZ.pkl' % dname
    with open(fname, 'rb') as f:
        rho_geo = pickle.load(f)

    fname = '%s/Bp.pkl' % dname
    with open(fname, 'rb') as f:
        Bp_geo = pickle.load(f)

    # Find Bp along outboard midplane and interpolate
    # onto desired computational grid
    ind = np.where(rho_geo == np.min(rho_geo))
    Bp = np.interp(
        rho,
        np.squeeze(rho_geo[ind[0], ind[1]:]),
        np.squeeze(Bp_geo[ind[0], ind[1]:]))

    # Unit conversions
    R0 = 1.78
    a = 0.56
    R = R0 + (a * rho)
    vtor = trot * R             # [vtor] = m / s
    vtor_err = trot_err * R     # [vtor] = m / s
    ne *= 1e19                  # [ne] = m^{-3}
    ne_err *= 1e19
    nep_err *= (1e19 / a)
    te *= 1e3                   # [te] = eV
    te_err *= 1e3
    tep_err *= (1e3 / a)

    Zeff = 2

    dndr = np.gradient(ne) / (a * np.gradient(rho))

    term1_pre = (dndr * te / ne) ** 2
    term1a = (ne_err / ne) ** 2
    term1b = (nep_err / dndr) ** 2
    term1c = (te_err / te) ** 2
    term1 = term1_pre * (term1a + term1b + term1c)

    term2 = (tep_err / Zeff) ** 2
    term3 = (vtor_err * Bp) ** 2

    Er_err = np.sqrt(term1 + term2 + term3)

    rho, Er = get_Er_force_balance(shot)

    Er_relerr = Er_err / Er

    if rhointerp is not None:
        Er_relerr = np.interp(
            rhointerp,
            rho,
            Er_relerr)
        rho = rhointerp

    return rho, Er_relerr


def get_dEr_uncertainty_force_balance(shot, rhointerp=None, Nsmooth=5):
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
    skiprows = 5

    # Load density data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'ne', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    rho = d[:, 0]   # radial coordinate
    ne = d[:, 1]    # [ne] = 10^{19} m^{-3}
    ne_err = d[:, 2]
    nep_err = d[:, 3]
    nep2_err = d[:, 4]

    # Load temperature data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'te', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    te = d[:, 1]    # [te] = keV
    te_err = d[:, 2]
    tep_err = d[:, 3]
    tep2_err = d[:, 4]

    # Load toroidal rotation data
    fname = '%s/d%s.xy.%i.0%i' % (dname, 'trot', shot, gaproftime)
    d = np.loadtxt(fname, skiprows=skiprows)
    trot = d[:, 1]  # [trot] = rad / s
    trot_err = d[:, 2]
    trotp_err = d[:, 3]

    # Load poloidal-field data
    dname = '../doppler_shift/%i/%ibis/geometry' % (shot, dirtime)

    fname = '%s/RHORZ.pkl' % dname
    with open(fname, 'rb') as f:
        rho_geo = pickle.load(f)

    fname = '%s/Bp.pkl' % dname
    with open(fname, 'rb') as f:
        Bp_geo = pickle.load(f)

    # Find Bp along outboard midplane and interpolate
    # onto desired computational grid
    ind = np.where(rho_geo == np.min(rho_geo))
    Bp = np.interp(
        rho,
        np.squeeze(rho_geo[ind[0], ind[1]:]),
        np.squeeze(Bp_geo[ind[0], ind[1]:]))

    # Unit conversions
    R0 = 1.78
    a = 0.56
    R = R0 + (a * rho)
    vtor = trot * R                 # [vtor] = m / s
    vtor_err = trot_err * R
    vtorp_err = trot_err * (R / a)
    ne *= 1e19                      # [ne] = m^{-3}
    ne_err *= 1e19
    nep_err *= (1e19 / a)
    nep2_err *= (1e19 / (a ** 2))
    te *= 1e3                       # [te] = eV
    te_err *= 1e3
    tep_err *= (1e3 / a)
    tep2_err *= (1e3 / (a ** 2))

    Zeff = 2

    dndr = np.gradient(ne) / (a * np.gradient(rho))
    d2ndr2 = np.gradient(dndr) / (a * np.gradient(rho))
    dtdr = np.gradient(te) / (a * np.gradient(rho))
    d2tdr2 = np.gradient(dtdr) / (a * np.gradient(rho))

    term1_pre = (((dndr ** 2) * te) / (Zeff * (ne ** 2))) ** 2
    term1a = (2 * ne_err / ne) ** 2
    term1b = (2 * nep_err / dndr) ** 2
    term1c = (te_err / te) ** 2
    term1 = term1_pre * (term1a + term1b + term1c)

    term2_pre = (d2ndr2 * te / (Zeff * ne)) ** 2
    term2a = (ne_err / ne) ** 2
    term2b = (nep2_err / d2ndr2) ** 2
    term2c = (te_err / te) ** 2
    term2 = term2_pre * (term2a + term2b + term2c)

    term3_pre = (dndr * dtdr / (Zeff * ne)) ** 2
    term3a = (ne_err / ne) ** 2
    term3b = (nep_err / dndr) ** 2
    term3c = (tep_err / dtdr) ** 2
    term3 = term3_pre * (term3a + term3b + term3c)

    term4 = tep2_err ** 2
    term5 = (vtorp_err * Bp) ** 2

    dEr_err = np.sqrt(term1 + term2 + term3 + term4 + term5)

    rho, Er = get_Er_force_balance(shot)
    dEr = np.gradient(Er) / (a * np.gradient(rho))

    dEr_relerr = np.abs(dEr_err / dEr)

    if rhointerp is not None:
        dEr_relerr = np.interp(
            rhointerp,
            rho,
            dEr_relerr)
        rho = rhointerp

    if Nsmooth is not None:
        dEr_relerr = np.convolve(
            dEr_relerr,
            np.ones(Nsmooth, dtype='float') / Nsmooth,
            mode='same')

    return rho, dEr_relerr


if __name__ == '__main__':
    shot = 171536
    Nsmooth = 10
    rho, dEr_relerr = get_dEr_uncertainty_force_balance(shot, Nsmooth=Nsmooth)

    plt.figure()
    plt.plot(rho, dEr_relerr)
    plt.show()
