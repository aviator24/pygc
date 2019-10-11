import numpy as np
import pandas as pd
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

def mask_ring(dat, mf_crit=0.9, Rmax=180):
    """mask ring by applying density threshold and radius cut"""
    rhoth = 0
    rho_mask = True
    R_mask = True

    if Rmax:
        R_mask = dat.R < Rmax

    if mf_crit:
        if Rmax:
            dat = dat.where(R_mask, other=0)
        rho, mf = _get_cummass(dat)
        rhoth = rho[np.searchsorted(-mf, -mf_crit)]
        rho_mask = dat.density > rhoth

    mask = rho_mask & R_mask
    return rhoth, mask

def count_SNe(s, dat, ts, te, ncrit):
    """Count the number of SNe exploded between time ts and te.

    Parameters
    ----------
    s     : LoadSim instance
    dat   : xr DataArray
    ts    : start time in code unit
    te    : end time in code unit
    ncrit : SNe exploded at hydrogen number density below ncrit is not counted.
    """
    le1, le2 = s.domain['le'][0], s.domain['le'][1]
    dx1, dx2 = s.domain['dx'][0], s.domain['dx'][1]
    sn = s.read_sn()[['time','x1sn','x2sn','navg']]
    sn = sn[(sn.time > ts)&(sn.time < te)&(sn.navg > ncrit)]
    sn['i'] = np.floor((sn.x1sn-le1)/dx1).astype('int32')
    sn['j'] = np.floor((sn.x2sn-le2)/dx2).astype('int32')
    sn = sn.groupby(['j','i']).size()
    # make Nx*Ny grid and project pSNe onto it
    i = np.arange(s.domain['Nx'][0])
    j = np.arange(s.domain['Nx'][1])
    idx = pd.MultiIndex.from_product([j,i], names=['j','i'])
    NSNe = pd.Series(np.zeros(s.domain['Nx'][0]*s.domain['Nx'][1]),
            index=idx)
    NSNe[sn.index] = sn
    NSNe = NSNe.unstack().values
    dat['NSNe'] = xr.DataArray(NSNe, dims=['y','x'],
            coords=[dat.coords['y'], dat.coords['x']])

def _Mabove(dat, rho_th):
    """Return total gas mass above threshold density rho_th."""
    rho = dat.density
    M = rho.where(rho>rho_th).sum()*dat.domain['dx'].prod()
    return M.values[()]

def _get_cummass(dat):
    """Return n_th and M(n>n_th)"""
    nbins = 50
    thresholds = np.linspace(0, 100, nbins)
    cummass = np.zeros(nbins)
    Mtot = _Mabove(dat, 0)
    for i, rho_th in enumerate(thresholds):
        cummass[i] = _Mabove(dat, rho_th) / Mtot
    return thresholds, cummass
