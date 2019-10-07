from pyathena.classic.cooling import coolftn
from pygc.derived_fields import add_derived_fields
from pygc.pot import gz_ext
import xarray as xr
import pandas as pd
import numpy as np
import pickle

def dataset_tavg(s, ts, te, Twarm=2.0e4, Rmax=180):
    """
    Function to time-average Datasets

    Parameters
    ----------
    s : LoadSimTIGRESSGC
        LoadSimTIGRESSGC instance to be analyzed.
    ts : int
        Start index.
    te : int
        End index.
    Twarm : float
        A demarcation temperature for two-phase medium
    Rmax : float
        A radius cut to mask out the nozzle

    Return
    ------
    dat : Dataset
        time-averaged Dataset
    """

    cf = coolftn()
    nums = np.arange(ts, te+1)
    sn = s.read_sn()[['time','x1sn','x2sn','x3sn']]
    tp = s.load_vtk(num=ts-1).domain['time']*s.u.Myr
    fields = ['density','velocity','pressure','gravitational_potential']
    time, Pdrive, Pturb, W = [], [], [], []

    # load a first vtk
    ds = s.load_vtk(num=ts)
    time.append(s.domain['time']*s.u.Myr)
    dat = ds.get_field(fields, as_xarray=True)
    add_derived_fields(s, dat, ['T','R'])
    tmp = dat.where((dat.T < Twarm)&(dat.R < Rmax), other=1e-15) # two-phase medium
    dat = xr.concat([dat,tmp], pd.Index(['all', '2p'], name='phase'))
    dat = dat.drop('T')

    Pdrive.append(_Pdrive(dat, sn, tp, time[-1]))
    Pturb.append(_Pturb(s, dat))
    W.append(_W(s,dat))

    # loop through vtks
    for num in nums[1:]:
        ds = s.load_vtk(num=num)
        time.append(ds.domain['time']*s.u.Myr)
        tmp = ds.get_field(fields, as_xarray=True)
        add_derived_fields(s, tmp, ['T','R'])
        tmp2 = tmp.where((tmp.T < Twarm)&(tmp.R < Rmax), other=1e-15)
        tmp = xr.concat([tmp,tmp2], pd.Index(['all', '2p'], name='phase'))
        tmp = tmp.drop('T')

        Pdrive.append(_Pdrive(tmp, sn, tp, time[-1]))
        Pturb.append(_Pturb(s, tmp))
        W.append(_W(s,tmp))
        
        # combine
        dat += tmp
    dat /= len(nums)
    dat['Pdrive'] = xr.DataArray(np.array(Pdrive, dtype=np.float32).T,
            dims=['phase','t'], coords={'phase':['all','2p'],'t':time})
    dat['Pturb'] = xr.DataArray(np.array(Pturb, dtype=np.float32).T,
            dims=['phase','t'], coords={'phase':['all','2p'],'t':time})
    dat['W'] = xr.DataArray(np.array(W, dtype=np.float32).T,
            dims=['phase','t'], coords={'phase':['all','2p'],'t':time})
    return dat

def _Pdrive(dat, sn, ts, te):
    dt = te-ts
    sn = sn[(sn.time > ts)&(sn.time < te)]
    n0 = dat.density.interp(z=0).mean(dim=['y','x']).values
    # SNe in two-phase cloudy medium; eqn 34 in Kim & Ostriker (2015)
    pstar = 2.8e5 * n0**-0.17
    NSNe = len(sn)
    return 0.25*pstar*NSNe/dt

def _Pturb(s, dat):
    dx = s.domain['dx'][0]
    dy = s.domain['dx'][1]
    Pturb = (dat.density*dat.velocity3**2).interp(z=0)
    return (Pturb.sum(dim=['y','x'])*dx*dy).values[()]*s.u.Msun*s.u.kms/s.u.Myr

def _W(s, dat):
    dx = s.domain['dx'][0]
    dy = s.domain['dx'][1]
    dz = s.domain['dx'][2]
    add_derived_fields(s, dat, ['R','gz_sg'])
    gz = dat.gz_sg + gz_ext(dat.R, dat.z)
    return (dat.density*s.u.Msun*gz).where(dat.z>0).sum(dim=['z','y','x'])*dx*dy*dz
