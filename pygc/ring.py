from pygc.derived_fields import add_derived_fields
import numpy as np
import pandas as pd
import xarray as xr
import os

def get_area(dm):
    """return the area (pc^2) of the masked region"""
    return ((dm.surf>0).sum() / (dm.domain['Nx'][0]*dm.domain['Nx'][1])
            *(dm.domain['Lx'][0]*dm.domain['Lx'][1])).values[()]

def mask_ring_by_mass(dat, mf_crit=0.9, Rmax=180):
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

def surfstar(s, dat, num, mask, area):
    """return stellar surface density in the masked region"""
    le1, le2 = s.domain['le'][0], s.domain['le'][1]
    dx1, dx2 = s.domain['dx'][0], s.domain['dx'][1]
    sp = s.load_starpar_vtk(num)
    sp['i'] = np.floor((sp.x1-le1)/dx1).astype('int32')
    sp['j'] = np.floor((sp.x2-le2)/dx2).astype('int32')
    sp = sp.groupby(['j','i']).sum()
    # make Nx*Ny grid and project pSNe onto it
    i = np.arange(s.domain['Nx'][0])
    j = np.arange(s.domain['Nx'][1])
    idx = pd.MultiIndex.from_product([j,i], names=['j','i'])
    msp = pd.Series(np.nan*np.zeros(s.domain['Nx'][0]*s.domain['Nx'][1]),
            index=idx)
    msp[sp.index] = sp.mass
    msp = msp.unstack().values
    msp = xr.DataArray(msp, dims=['y','x'],
            coords=[dat.coords['y'], dat.coords['x']])
    msp = msp.where(mask).sum().values[()]
    return msp/area

def average_ring(s, num, Tmax=2e4, Rmax=200):
    ds = s.load_vtk(num)
    dx = ds.domain['dx'][0]
    dy = ds.domain['dx'][1]
    dat = ds.get_field(['density','velocity','pressure',
        'gravitational_potential'], as_xarray=True)
    dat = dat.drop(['velocity1','velocity2'])
    add_derived_fields(dat, ['R','T','gz_sg'], in_place=True)
    dat = dat.drop(['gravitational_potential'])
    gz_sg = dat.gz_sg
    dat = dat.where((dat.T<Tmax)&(dat.R<Rmax))
    dat = dat.drop(['T'])
    dat['gz_sg'] = gz_sg
    add_derived_fields(dat, ['Pturb','Pgrav'], in_place=True)
    dat = dat.drop(['density','velocity3','gz_sg'])
    dat = dat.sel(z=1)
    area = get_area(dat)
    t = ds.domain['time']*s.u.Myr
    Pth = (dat.pressure*dx*dy).sum().values[()]*s.u.pok/area
    Pturb = (dat.Pturb*dx*dy).sum().values[()]*s.u.pok/area
    Pgrav = (dat.Pgrav*dx*dy).sum().values[()]*s.u.pok/area
    return [t,Pth,Pturb,Pgrav,area]

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


if __name__ == '__main__':
    import argparse
    from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC

    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('Tmax', type=float)
    parser.add_argument('Rmax', type=float)
    parser.add_argument('-v', '--verbosity', action='count',
                        help='increase output verbosity')
    parser.add_argument('--outdir', default=None, help='output directory')
    parser.add_argument('--mpi', action='store_true', help='enable mpi')
    args = parser.parse_args()

    if args.mpi:
        from mpi4py import MPI
        from pyathena.util.split_container import split_container
        COMM = MPI.COMM_WORLD
        myrank = COMM.rank
    else:
        myrank = 0

    if args.outdir==None:
        outdir=args.indir+'/timeseries'
    if (~os.path.exists(outdir))&(myrank==0):
        os.mkdir(outdir)

    nums = np.arange(args.start,args.end+1)
    if args.mpi:
        if myrank == 0:
            nums = split_container(nums, COMM.size)
        else:
            nums = None
        mynums = COMM.scatter(nums, root=0)
        print('[rank, mysteps]:', myrank, mynums)
    else:
        mynums=nums

    # load simulation and perform local time-average
    s = LoadSimTIGRESSGC(args.indir)

    for num in mynums:
        ds = average_ring(s, num, Tmax=args.Tmax, Rmax=args.Rmax)
        np.savetxt("{}/ts.{:.0f}.{:04d}.txt".format(outdir,args.Rmax,num), ds)
