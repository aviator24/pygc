from .pot import MHubble
from pyathena.util.units import Units
from pyathena.classic.cooling import coolftn
import numpy as np
import pandas as pd
import xarray as xr

Twarm = 2.0e4
u = Units()
extpot = MHubble(120, 265)

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

def add_derived_fields(dat, fields=[], in_place=True):
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
        if not 'Pturb' in dat.data_vars:
            add_derived_fields(dat, fields='Pturb', in_place=True)
        if in_place:
            dat['sz'] = np.sqrt((dat.Pturb/dat.density).interp(z=0))
        else:
            tmp['sz'] = np.sqrt((dat.Pturb/dat.density).interp(z=0))

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
        Pgrav = (dat.density*
                    (dat.gz_sg+extpot.gz(dat.x, dat.y, dat.z).T)
                ).where(dat.z>0).sum(dim='z')*dz
        if in_place:
            dat['Pgrav'] = -Pgrav
        else:
            tmp['Pgrav'] = -Pgrav

    if 'T' in fields:
        cf = coolftn()
        pok = dat.pressure*u.pok
        T1 = pok/(dat.density*u.muH) # muH = Dcode/mH
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

    if not in_place:
        return tmp

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

def grid_msp(s, num, agemin, agemax):
    """read starpar_vtk and remap starpar mass onto a grid"""
    # domain information
    le1, le2 = s.domain['le'][0], s.domain['le'][1]
    re1, re2 = s.domain['re'][0], s.domain['re'][1]
    dx1, dx2 = s.domain['dx'][0], s.domain['dx'][1]
    Nx1, Nx2 = s.domain['Nx'][0], s.domain['Nx'][1]
    i = np.arange(Nx1)
    j = np.arange(Nx2)
    x = np.linspace(le1+0.5*dx1, re1-0.5*dx1, Nx1)
    y = np.linspace(le2+0.5*dx2, re2-0.5*dx2, Nx2)
    # load starpar vtk
    sp = s.load_starpar_vtk(num)[['x1','x2','mass','mage']]
    # apply age cut
    sp = sp[(sp['mage'] < agemax)&
            (sp['mage'] > agemin)]
    # remap the starpar onto a grid
    sp['i'] = np.floor((sp.x1-le1)/dx1).astype('int32')
    sp['j'] = np.floor((sp.x2-le2)/dx2).astype('int32')
    sp = sp.groupby(['j','i']).sum()
    idx = pd.MultiIndex.from_product([j,i], names=['j','i'])
    msp = pd.Series(np.zeros(Nx1*Nx2), index=idx)
    msp[sp.index] = sp.mass
    msp = msp.unstack().values
    msp = xr.DataArray(msp, dims=['y','x'], coords=[y,x])
    return msp

def sum_dataset(s, nums, twophase=False):
    """Do time-summation on Datasets and return the summed Dataset.

    Parameters
    ----------
    s : LoadSimTIGRESSGC instance to be analyzed.
    nums :list of snapshot numbers to add
    twophase : include two-phase gas only
    """

    fields = ['density','velocity','pressure','gravitational_potential']

    # load a first vtk
    ds = s.load_vtk(num=nums[0])
    dat = ds.get_field(fields, as_xarray=True)
    if twophase:
        Phi = dat.gravitational_potential
        add_derived_fields(dat, fields='T', in_place=True)
        dat = dat.where(dat.T < Twarm, other=0)
        dat = dat.drop('T')
        dat['gravitational_potential'] = Phi
    add_derived_fields(dat, fields=['R','Pturb','Pgrav'], in_place=True)

    # loop through vtks
    for num in nums[1:]:
        ds = s.load_vtk(num=num)
        tmp = ds.get_field(fields, as_xarray=True)
        if twophase:
            Phi = tmp.gravitational_potential
            add_derived_fields(tmp, fields='T', in_place=True)
            tmp = tmp.where(tmp.T < Twarm, other=0)
            tmp = tmp.drop('T')
            tmp['gravitational_potential'] = Phi
        add_derived_fields(tmp, fields=['R','Pturb','Pgrav'], in_place=True)
        # add
        dat += tmp
    return dat

def read_stardat(fpath, num):
    ds = np.loadtxt("{}/star{:05d}.dat".format(fpath, num))
    return {'t':ds[:,0], 'm':ds[:,1], 'x1':ds[:,2], 'x2':ds[:,3], 'x3':ds[:,4],
            'v1':ds[:,5], 'v2':ds[:,6], 'v3':ds[:,7], 'age':ds[:,8],
            'mage':ds[:,9], 'mdot':ds[:,10], 'merge_history':ds[:,11]}

def read_ring(indir, ns, ne, mf_crit=False, twophase=False):
    t, surf, surfstar, surfsfr, n0, H, Hs, sz, Pgrav_gas, Pgrav_starpar, Pgrav_ext, \
    Pturb, Pth, area = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    nums = np.arange(ns, ne+1)
    fname = 'gc'
    if twophase:
        fname = fname+'.2p'
    if mf_crit:
        fname = fname+'.mcut{}'.format(mf_crit)
    for num in nums:
        try:
            ds = np.loadtxt("{}/{}.{:04d}.txt".format(indir, fname, num))
            t.append(ds[0])
            surf.append(ds[1])
            surfstar.append(ds[2])
            surfsfr.append(ds[3])
            n0.append(ds[4])
            H.append(ds[5])
            Hs.append(ds[6])
            sz.append(ds[7])
            Pgrav_gas.append(ds[8])
            Pgrav_starpar.append(ds[9])
            Pgrav_ext.append(ds[10])
            Pturb.append(ds[11])
            Pth.append(ds[12])
            area.append(ds[13])
        except OSError:
            pass
    t = np.array(t)*u.Myr
    surf = np.array(surf)*u.Msun
    surfstar = np.array(surfstar)*u.Msun
    surfsfr = np.array(surfsfr)*u.Msun/u.Myr
    n0 = np.array(n0)
    H = np.array(H)
    Hs = np.array(Hs)
    sz = np.array(sz)
    Pgrav_gas = np.array(Pgrav_gas)*u.pok
    Pgrav_starpar = np.array(Pgrav_starpar)*u.pok
    Pgrav_ext = np.array(Pgrav_ext)*u.pok
    Pturb = np.array(Pturb)*u.pok
    Pth = np.array(Pth)*u.pok
    return {'t':t, 'surf':surf, 'surfstar':surfstar, 'surfsfr':surfsfr, 'n0':n0,
            'H':H, 'Hs':Hs, 'sz':sz, 'Pgrav_gas':Pgrav_gas,
            'Pgrav_starpar':Pgrav_starpar, 'Pgrav_ext':Pgrav_ext, 
            'Pturb':Pturb, 'Pth':Pth}

