from pyathena.classic.cooling import coolftn
from pygc.derived_fields import add_derived_fields
from pygc.pot import gz_ext
import xarray as xr
import pandas as pd
import numpy as np
import pickle

def dataset_tavg(s, ts, te, Twarm=2.0e4):
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

    Return
    ------
    dat : Dataset
        time-averaged Dataset
    """

    cf = coolftn()
    nums = np.arange(ts, te+1)
    fields = ['density','velocity','pressure','gravitational_potential']

    # load a first vtk
    ds = s.load_vtk(num=ts)
    dat = ds.get_field(fields, as_xarray=True)
    add_derived_fields(dat, fields=['T','R'], in_place=True)
    tmp = dat.where(dat.T < Twarm, other=1e-15) # two-phase medium
    dat = xr.concat([dat,tmp], pd.Index(['all', '2p'], name='phase'))
    dat = dat.drop('T')
    add_derived_fields(dat, fields='Pturb', in_place=True)

    # loop through vtks
    for num in nums[1:]:
        ds = s.load_vtk(num=num)
        tmp = ds.get_field(fields, as_xarray=True)
        add_derived_fields(tmp, fields=['T','R'], in_place=True)
        tmp2 = tmp.where(tmp.T < Twarm, other=1e-15)
        tmp = xr.concat([tmp,tmp2], pd.Index(['all', '2p'], name='phase'))
        tmp = tmp.drop('T')
        add_derived_fields(tmp, fields='Pturb', in_place=True)
        
        # combine
        dat += tmp
    dat /= len(nums)
    return dat
