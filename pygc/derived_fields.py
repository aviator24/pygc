from pyathena.classic.cooling import coolftn
from .xrutil import wmean
from pyathena.util.units import Units
import numpy as np
import xarray as xr

#def mask_sn(sne, mask):
#    msne = pd.DataFrame()
#    for i in range(len(sne)):
#        sn = sne.iloc[i]
#        ip,jp,kp = cc_ijk(sn.x1sn, sn.x2sn, sn.x3sn)
#        if mask[kp,jp,ip]:
#            msne = msne.append(sn)

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
    sn = [(sn.time > dat.ts)&(sn.time < dat.te)]
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
    u = Units()

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
