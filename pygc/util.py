import numpy as np
import xarray as xr
def wmean(arr, weights, dim):
    """
    Function to compute weighted mean of xarray.

    Parameters
    ----------
    arr : DataArray
        Input array to calculate weighted mean. May contain NaNs.
    weights : DataArray
        Array containing weights.
    dim : 'x', 'y', or 'z'
        xarray dimension to perform weighted mean.

    Return
    ------
    (\int arr * w) / (\int w)

    """
    if type(arr) is not xr.core.dataarray.DataArray:
        raise Exception("only xarray type is supported!\n")
    else:
        notnull = arr.notnull()
        return (arr * weights).sum(dim=dim) / weights.where(notnull).sum(dim=dim)

def _Mabove(dat, rho_th):
    """Return total gas mass above threshold density rho_th."""
    rho = dat.density
    M = rho.where(rho>rho_th).sum()*dat.domain['dx'].prod()
    return M.values[()]

def _get_cummass(dat):
    """Return n_th and M(n>n_th)"""
    nbins = 100
    thresholds = np.logspace(0, np.log10(dat.density.max().values[()]), nbins)
    cummass = np.zeros(nbins)
    Mtot = _Mabove(dat, 0)
    for i, rho_th in enumerate(thresholds):
        cummass[i] = _Mabove(dat, rho_th) / Mtot
    return thresholds, cummass

def mask_ring(dat, mf_crit=0.9, Rmax=180):
    """mask ring by applying density threshold and radius cut"""
    rho, mf = _get_cummass(dat)
    rhoth = rho[np.searchsorted(-mf, -mf_crit)]
    mask = (dat.density > rhoth)&(dat.R < Rmax)
    return rhoth, mask
