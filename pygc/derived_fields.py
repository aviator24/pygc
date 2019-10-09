from pyathena.classic.cooling import coolftn
from .util import wmean
from pyathena.util.units import Units
import numpy as np
import pandas as pd
import xarray as xr

u = Units()
def set_Pdrive(s, dat):
    """Return momentum injection rate per area (pressure) from SNe."""

    le1, le2 = s.domain['le'][0], s.domain['le'][1]
    dx1, dx2 = s.domain['dx'][0], s.domain['dx'][1]
    sn = s.read_sn()[['time','x1sn','x2sn','navg']]
    sn = sn[(sn.time > dat.ts)&(sn.time < dat.te)]
    sn['i'] = np.floor((sn.x1sn-le1)/dx1).astype('int32')
    sn['j'] = np.floor((sn.x2sn-le2)/dx2).astype('int32')
    sn['pSNe'] = 2.8e5*sn['navg']**-0.17
    sn = sn.groupby(['j','i']).sum()
    # make Nx*Ny grid and project pSNe onto it
    i = np.arange(s.domain['Nx'][0])
    j = np.arange(s.domain['Nx'][1])
    idx = pd.MultiIndex.from_product([j,i], names=['j','i'])
    pSNe = pd.Series(np.zeros(s.domain['Nx'][0]*s.domain['Nx'][1]),
            index=idx)
    pSNe[sn.index] = sn.pSNe
    pSNe = pSNe.unstack().values

    pSNe = xr.DataArray(pSNe, dims=['y','x'],
            coords=[dat.coords['y'], dat.coords['x']])
    Pdrive = 0.25*pSNe/(dat.te-dat.ts)/dx1/dx2/(s.u.Msun/s.u.Myr)
    dat['Pdrive'] = Pdrive

def dpdt_sn(s, dat):
    """Return vertical momentum injection rate from SNe during ts-te.

    Parameters
    ----------
    s   : LoadSim instance
    dat : Time-averaged Dataset

    Notes
    ----
    Pdrive = dpdt_sn / area
    """

    sn = s.read_sn()[['time','x1sn','x2sn','x3sn']]
    sn = sn[(sn.time > dat.ts)&(sn.time < dat.te)]
    NSNe = len(sn)
    n0 = dat.density.interp(z=0).mean().values
    pstar = 2.8e5*n0**-0.17 # Kim & Ostriker, Eqn. (34)
    return 0.25*pstar*NSNe/(dat.te-dat.ts)

def add_derived_fields(dat, fields=[], in_place=False):
    """Add derived fields in a Dataset

    Parameters
    ----------
    dat    : xarray Dataset of variables
    fields : list containing derived fields to be added.
               ex) ['H', 'surf', 'T']
    """

    dx = (dat.x[1]-dat.x[0]).values[()]
    dy = (dat.y[1]-dat.y[0]).values[()]
    dz = (dat.z[1]-dat.z[0]).values[()]

    if not in_place:
        tmp = dat.copy()

    if 'H' in fields:
        zsq = (dat.z.where(~np.isnan(dat.density)))**2
        H2 = wmean(zsq, dat.density, 'z')
        if in_place:
            dat['H'] = np.sqrt(H2)
        else:
            tmp['H'] = np.sqrt(H2)

    if 'surf' in fields:
        if in_place:
            dat['surf'] = (dat.density*dz).sum(dim='z')
        else:
            tmp['surf'] = (dat.density*dz).sum(dim='z')
        
    if 'sz' in fields:
        if in_place:
            dat['sz'] = np.sqrt(dat.velocity3.interp(z=0)**2)
        else:
            tmp['sz'] = np.sqrt(dat.velocity3.interp(z=0)**2)

    if 'R' in fields:
        if in_place:
            dat.coords['R'] = np.sqrt(dat.x**2 + dat.y**2)
        else:
            tmp.coords['R'] = np.sqrt(dat.x**2 + dat.y**2)

    if 'Pturb' in fields:
        if in_place:
            dat['Pturb'] = dat.density*dat.velocity3**2
        else:
            tmp['Pturb'] = dat.density*dat.velocity3**2

    if 'Pgrav' in fields:
        if not 'gz_sg' in dat.data_vars:
            add_derived_fields(dat, fields='gz_sg', in_place=True)
        Pgrav = (dat.density*dat.gz_sg).where(dat.z>0).sum(dim='z')*dz
        if in_place:
            dat['Pgrav'] = Pgrav
        else:
            tmp['Pgrav'] = Pgrav

    if 'T' in fields:
        cf = coolftn()
        pok = dat.pressure*u.pok
        T1 = pok/dat.density*u.muH
        if in_place:
            dat['T'] = xr.DataArray(cf.get_temp(T1.values), coords=T1.coords,
                    dims=T1.dims)
        else:
            tmp['T'] = xr.DataArray(cf.get_temp(T1.values), coords=T1.coords,
                    dims=T1.dims)

    if 'gz_sg' in fields:
        phir = dat.gravitational_potential.shift(z=-1)
        phil = dat.gravitational_potential.shift(z=1)
        phir.loc[{'z':phir.z[-1]}] = 3*phir.isel(z=-2) - 3*phir.isel(z=-3) + phir.isel(z=-4)
        phil.loc[{'z':phir.z[0]}] = 3*phil.isel(z=1) - 3*phil.isel(z=2) + phil.isel(z=3)
        if in_place:
            dat['gz_sg'] = (phil-phir)/dz
        else:
            tmp['gz_sg'] = (phil-phir)/dz

    if in_place:
        return True
    else:
        return tmp
