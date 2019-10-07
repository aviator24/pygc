from pyathena.classic.cooling import coolftn
from pyathena.util.wmean import wmean
import numpy as np
import xarray as xr
def add_derived_fields(s, dat, fields, in_place=True):
    """
    Function to add derived fields in DataSet.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance for your simulation (e.g., LoadSimTIGRESSGC).
    dat : Dataset
        xarray Dataset of variables
    fields : list
        list containing derived fields to be added.
        ex) ['H', 'surf', 'T']

    Return
    ------
    dat : Dataset
        new xarray Dataset with derived_field
    """
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
            dat['surf'] = (dat.density*s.domain['dx'][2]).sum(dim='z')
        else:
            tmp['surf'] = (dat.density*s.domain['dx'][2]).sum(dim='z')
        
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

    if 'T' in fields:
        cf = coolftn()
        pok = dat.pressure*s.u.pok
        T1 = pok/dat.density*s.u.muH
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
            dat['gz_sg'] = (phil-phir)/s.domain['dx'][2]
        else:
            tmp['gz_sg'] = (phil-phir)/s.domain['dx'][2]

    if in_place:
        return dat
    else:
        return tmp
