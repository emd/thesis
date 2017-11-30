import numpy as np


def wavenumber(f_kHz, cs=343.):
    '''Return wavenumber of sound wave in cm^{-1}.

    Parameters:
    -----------
    f_kHz - float
        Frequency of sound wave.
        [f_kHz] = kHz

    cs - float
        Sound speed.
        [cs] = m/s

    Returns:
    --------
    k - float
        Wavenumber of sound wave
        [k] = cm^{-1}

    '''
    # Convert frequency to Hz and get wavenumber in m^{-1}
    k = 2 * np.pi * (f_kHz * 1e3) / cs

    # Convert to cm^{-1}
    k *= 1e-2

    return k
