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

def count_SNe(s, ts, te, ncrit):
    """Count the number of SNe and map the result onto a grid

    Parameters
    ----------
    s     : LoadSim instance
    ts    : start time in code unit
    te    : end time in code unit
    ncrit : SNe exploded at hydrogen number density below ncrit is not counted.

    Return
    ------
    NSNe : xr.DataArray of the number of supernovae
    """
    # domain information
    le1, le2 = s.domain['le'][0], s.domain['le'][1]
    re1, re2 = s.domain['re'][0], s.domain['re'][1]
    dx1, dx2 = s.domain['dx'][0], s.domain['dx'][1]
    Nx1, Nx2 = s.domain['Nx'][0], s.domain['Nx'][1]
    i = np.arange(Nx1)
    j = np.arange(Nx2)
    x = np.linspace(le1+0.5*dx1, re1-0.5*dx1, Nx1)
    y = np.linspace(le2+0.5*dx2, re2-0.5*dx2, Nx2)
    # load supernova dump
    sn = s.read_sn()[['time','x1sn','x2sn','navg']]
    # filter SNs satisfying (ts < t < te) and (n > n_crit)
    sn = sn[(sn.time > ts)&(sn.time < te)&(sn.navg > ncrit)]
    # remap the number of SNs onto a grid
    sn['i'] = np.floor((sn.x1sn-le1)/dx1).astype('int32')
    sn['j'] = np.floor((sn.x2sn-le2)/dx2).astype('int32')
    sn = sn.groupby(['j','i']).size()
    idx = pd.MultiIndex.from_product([j,i], names=['j','i'])
    NSNe = pd.Series(np.nan*np.zeros(Nx1*Nx2), index=idx)
    NSNe[sn.index] = sn
    NSNe = NSNe.unstack().values
    NSNe = xr.DataArray(NSNe, dims=['y','x'], coords=[y, x])
    return NSNe
