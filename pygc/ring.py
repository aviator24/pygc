from .util import add_derived_fields, count_SNe
import numpy as np
import pandas as pd
import xarray as xr
import os
from scipy.optimize import bisect

Twarm=2.e4

def do_average(s, num, twophase=True, pgravmask=True):
    """Load num-th snapshot, define ring region, and do midplane average"""
    ds = s.load_vtk(num)
    dx = ds.domain['dx'][0]
    dy = ds.domain['dx'][1]
    dat = ds.get_field(['density', 'velocity', 'pressure',
        'gravitational_potential'], as_xarray=True)
    dat = dat.drop(['velocity1','velocity2'])
    if twophase:
        add_derived_fields(dat, ['T','gz_sg'], in_place=True)
        gz_sg = dat.gz_sg
        dat = dat.where((dat.T<Twarm))
        dat = dat.drop(['T'])
        dat['gz_sg'] = gz_sg
    else:
        add_derived_fields(dat, ['gz_sg'], in_place=True)
    dat = dat.drop(['gravitational_potential'])
    # Now, dat contains [density, velocity3, pressure, gz_sg]

    # calculate total weight
    add_derived_fields(dat, ['Pgrav'], in_place=True)
    dat = dat.drop(['gz_sg'])
    dat['density'] = dat.density.sel(z=0, method='nearest')
    dat['velocity3'] = dat.velocity3.sel(z=0, method='nearest')
    dat['pressure'] = dat.pressure.sel(z=0, method='nearest')
    add_derived_fields(dat, ['Pturb'], in_place=True)
    dat = dat.drop(['density','velocity3','z'])
    if pgravmask:
        dat['Pgrav'] = dat.Pgrav.where(dat.Pturb>0)
    t = ds.domain['time']
    area = _get_area(dat)
    Pth = (dat.pressure*dx*dy).sum().values[()]/area
    Pturb = (dat.Pturb*dx*dy).sum().values[()]/area
    Pgrav = (dat.Pgrav*dx*dy).sum().values[()]/area
    return [t,Pth,Pturb,Pgrav,area]

def mask_ring_by_mass(dat, mf_crit=0.9, Rmax=None):
    """mask ring by applying density threshold and radius cut"""
    mask = True
    R_mask = True

    if Rmax:
        if not 'R' in dat.data_vars:
            add_derived_fields(dat, fields='R', in_place=True)
        R_mask = dat.R < Rmax

    if mf_crit:
        if Rmax:
            dat = dat.where(R_mask, other=0)
        Mtot = _Mabove(dat, 0)
        surf_th = bisect(lambda x: mf_crit*Mtot-_Mabove(dat, x), 1e1, 1e5)
        mask = dat.surf > surf_th

    mask = mask & R_mask
    return surf_th, mask

def _get_area(dm):
    """return the area (pc^2) of the masked region"""
    if 'Pturb' in dm.data_vars:
        area = ((dm.Pturb>0).sum() / (dm.domain['Nx'][0]*dm.domain['Nx'][1])
            *(dm.domain['Lx'][0]*dm.domain['Lx'][1])).values[()]
    elif 'surf' in dm.data_vars:
        area = ((dm.surf>0).sum() / (dm.domain['Nx'][0]*dm.domain['Nx'][1])
            *(dm.domain['Lx'][0]*dm.domain['Lx'][1])).values[()]
    else:
        raise ValueError("input data should contain Pturb or surf field")
    return area

def _Mabove(dat, surf_th):
    """Return total gas mass above threshold density surf_th."""
    surf = dat.surf
    M = surf.where(surf>surf_th).sum()*dat.domain['dx'][0]*dat.domain['dx'][1]
    return M.values[()]
