from .util import add_derived_fields, grid_msp
from .pot import MHubble, Plummer
from pyathena.io.read_vtk import read_vtk
from scipy.optimize import bisect
import numpy as np

Twarm = 2.0e4

def mask_ring_by_mass(dat, mf_crit=0.9, Rmax=None):
    """mask ring by applying density threshold and radius cut"""
    mask = True
    R_mask = True
    surf_th = 0

    if Rmax:
        if not 'R' in dat.data_vars:
            add_derived_fields(dat, fields='R')
        R_mask = dat.R < Rmax

    if mf_crit:
        if Rmax:
            dat = dat.where(R_mask, other=0)
        Mtot = _Mabove(dat, 0)
        surf_th = bisect(lambda x: mf_crit*Mtot-_Mabove(dat, x), 1e1, 1e4)
        mask = dat.surf > surf_th

    mask = mask & R_mask
    return surf_th, mask

def ring_avg(s, num, mask, sfr_dt=10):
    dz = s.domain['dx'][2]
    bul = MHubble(s.par['problem']['R_b'], s.par['problem']['rho_b'])
    BH = Plummer(s.par['problem']['M_c'], s.par['problem']['R_c'])

    ds = s.load_vtk(num)
    sp = s.load_starpar_vtk(num)
    if sp.shape[1]==0:
        print("no star particles are found")
        flag_sp = False
    else:
        flag_sp = True
    t = ds.domain['time']
    dat = ds.get_field(['density','velocity','pressure',
        'gravitational_potential'], as_xarray=True)
    dat = dat.drop(['velocity1','velocity2'])
    add_derived_fields(dat, 'gz_sg')
    dat = dat.drop('gravitational_potential')

    # seperate individual contributions to the gravitational field.
    ds = read_vtk('{}/postproc_gravity/gc.{:04d}.Phi.vtk'.format(s.basedir, num))
    Phigas = ds.get_field('Phi').Phi
    phir = Phigas.shift(z=-1)
    phil = Phigas.shift(z=1)
    phir.loc[{'z':phir.z[-1]}] = 3*phir.isel(z=-2) - 3*phir.isel(z=-3) + phir.isel(z=-4)
    phil.loc[{'z':phir.z[0]}] = 3*phil.isel(z=1) - 3*phil.isel(z=2) + phil.isel(z=3)
    gz_gas = (phil-phir)/(2*dz)
    dat['gz_starpar'] = dat.gz_sg - gz_gas # order is important!
    dat['gz_gas'] = gz_gas # order is important!
    dat['gz_ext'] = bul.gz(dat.x, dat.y, dat.z).T + BH.gz(dat.x, dat.y, dat.z).T

    # add derived fields
    
    dat['Wgas'] = 0.5*(dat.density*abs(dat.gz_gas)*dz).sum(dim='z')
    dat['Wsp'] = 0.5*(dat.density*abs(dat.gz_starpar)*dz).sum(dim='z')
    dat['Wext'] = 0.5*(dat.density*abs(dat.gz_ext)*dz).sum(dim='z')

    dat['Wgas_oneside'] = -(dat.density*dat.gz_gas    *dz).where(dat.z>0).sum(dim='z')
    dat['Wsp_oneside']  = -(dat.density*dat.gz_starpar*dz).where(dat.z>0).sum(dim='z')
    dat['Wext_oneside'] = -(dat.density*dat.gz_ext    *dz).where(dat.z>0).sum(dim='z')

    dat = dat.drop(['gz_sg', 'gz_starpar', 'gz_gas', 'gz_ext'])
    add_derived_fields(dat, ['surf','Pturb'])
    dat = dat.drop('velocity3')

    dat['n0'] = dat.density.interp(z=0)
    dat['Pth_mid'] = dat.pressure.interp(z=0)
    dat['Pturb_mid'] = dat.Pturb.interp(z=0)
    dat['Pth_top'] = 0.5*(dat.pressure.isel(z=-1)+dat.pressure.isel(z=0))
    dat['Pturb_top'] = 0.5*(dat.Pturb.isel(z=-1)+dat.Pturb.isel(z=0))

    area = _get_area(dat.where(mask))

    surf = dat.surf.where(mask).mean().values[()]
    if flag_sp:
        msp = grid_msp(s, num, 0, 1e10)
        surfstar = msp.where(mask).sum().values[()]/area
        agebin = sfr_dt/s.u.Myr
        msp = grid_msp(s, num, 0, agebin)
        surfsfr_ring = msp.where(mask).sum().values[()]/area/agebin
        surfsfr_whole = msp.sum().values[()]/area/agebin
    else:
        surfstar = 0
        surfsfr_ring = 0
        surfsfr_whole = 0
    Pth_mid = dat.Pth_mid.where(mask).mean().values[()]
    Pturb_mid = dat.Pturb_mid.where(mask).mean().values[()]
    Pth_top = dat.Pth_top.where(mask).mean().values[()]
    Pturb_top = dat.Pturb_top.where(mask).mean().values[()]
    Wgas = dat.Wgas.where(mask).mean().values[()]
    Wsp = dat.Wsp.where(mask).mean().values[()]
    Wext = dat.Wext.where(mask).mean().values[()]
    Wgas_oneside = dat.Wgas_oneside.where(mask).mean().values[()]
    Wsp_oneside = dat.Wsp_oneside.where(mask).mean().values[()]
    Wext_oneside = dat.Wext_oneside.where(mask).mean().values[()]
    n0 = dat.n0.where(mask).mean().values[()]

    return [t, area, surf, surfsfr_ring, surfsfr_whole, Pth_mid, Pturb_mid, Pth_top, Pturb_top, Wgas, Wsp, Wext, Wgas_oneside, Wsp_oneside, Wext_oneside, n0, surfstar]

def _get_area(dm):
    """return the area (pc^2) of the masked region"""
    if 'surf' in dm.data_vars:
        area = ((dm.surf>0).sum() / (dm.domain['Nx'][0]*dm.domain['Nx'][1])
            *(dm.domain['Lx'][0]*dm.domain['Lx'][1])).values[()]
    elif 'Pturb' in dm.data_vars:
        if len(dm.Pturb.dims)==3:
            raise ValueError("Pturb should be given as midplane value")
        area = ((dm.Pturb>0).sum() / (dm.domain['Nx'][0]*dm.domain['Nx'][1])
            *(dm.domain['Lx'][0]*dm.domain['Lx'][1])).values[()]
    else:
        raise ValueError("input data should contain Pturb or surf field")
    return area

def _Mabove(dat, surf_th):
    """Return total gas mass above threshold density surf_th."""
    surf = dat.surf
    M = surf.where(surf>surf_th).sum()*dat.domain['dx'][0]*dat.domain['dx'][1]
    return M.values[()]
