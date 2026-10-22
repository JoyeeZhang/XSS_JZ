import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

def tau(x):
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)

    small = np.abs(x) < 1e-12
    out[small] = 0.5

    out[~small] = np.sin(x[~small])/x[~small] + (np.cos(x[~small])-1)/x[~small]**2

    return out

def kernel_with_raised_cosine_window(ind_s, ds, alpha=0.54):
    factor = 0.5 / ds**2
    val = np.arange(-ind_s+1, ind_s)
    filter = factor * (
        alpha * tau(np.pi*val) +
        0.5 * (1-alpha) * tau(np.pi*(val+1)) +
        0.5 * (1-alpha) * tau(np.pi*(val-1))
    )

    return filter
