#!/usr/bin/env python
"""
=======================================================================
Description | Read snapshot, delineate ring, compute various quantities
Author      | Sanghyuk Moon
=======================================================================
"""
from pygc.util import add_derived_fields, grid_msp
from pygc.pot import gz_ext
from pygc.ring import mask_ring_by_mass, _get_area
from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
from pyathena.io.read_vtk import read_vtk
import argparse
import numpy as np
import os

Twarm = 2.e4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('mf_crit', type=float, help='mass cut')
    parser.add_argument('Rmax', type=float, help='radius cut')
    parser.add_argument('--outdir', default=None, help='output directory')
    parser.add_argument('--mpi', action='store_true', help='enable mpi')
    parser.add_argument('--twophase', action='store_true')
    args = parser.parse_args()

    if args.mpi:
        from mpi4py import MPI
        from pyathena.util.split_container import split_container
        COMM = MPI.COMM_WORLD
        myrank = COMM.rank
    else:
        myrank = 0

    if args.outdir==None:
        outdir=args.indir+'/postproc_ring'
    else:
        outdir=args.outdir
    if (~os.path.exists(outdir))&(myrank==0):
        os.mkdir(outdir)

    fname = 'gc'
    if args.twophase:
        fname = fname+'.2p'

    nums = np.arange(args.start,args.end+1)
    if args.mpi:
        if myrank == 0:
            nums = split_container(nums, COMM.size)
        else:
            nums = None
        mynums = COMM.scatter(nums, root=0)
    else:
        mynums=nums
    print('[rank, mysteps]:', myrank, mynums)

    # load simulation
    s = LoadSimTIGRESSGC(args.indir)
    dz = s.domain['dx'][2]

    for num in mynums:
        ds = s.load_vtk(num)
        sp = s.load_starpar_vtk(num)
        t = ds.domain['time']
        dat = ds.get_field(['density','velocity','pressure',
            'gravitational_potential'], as_xarray=True)
        dat = dat.drop(['velocity1','velocity2'])

        if args.twophase:
            add_derived_fields(dat, ['T','gz_sg'])
            gz_sg = dat.gz_sg
            # select two-phase gas
            dat = dat.where(dat.T < Twarm)
            dat = dat.drop('T')
            dat['gz_sg'] = gz_sg
        else:
            add_derived_fields(dat, ['gz_sg'])
        dat = dat.drop('gravitational_potential')

        # seperate individual contributions to the gravitational field.
        ds = read_vtk('{}/postproc_gravity/gc.{:04d}.vtk'.format(s.basedir, num))
        Phigas = ds.get_field('Phi').Phi
        phir = Phigas.shift(z=-1)
        phil = Phigas.shift(z=1)
        phir.loc[{'z':phir.z[-1]}] = 3*phir.isel(z=-2) - 3*phir.isel(z=-3) + phir.isel(z=-4)
        phil.loc[{'z':phir.z[0]}] = 3*phil.isel(z=1) - 3*phil.isel(z=2) + phil.isel(z=3)
        gz_gas = (phil-phir)/dz
        dat['gz_starpar'] = dat.gz_sg - gz_gas # order is important!
        dat['gz_gas'] = gz_gas # order is important!
        add_derived_fields(dat, 'R')
        dat['gz_ext'] = gz_ext(dat.R, dat.z, TIGRESS_unit=True).T

        # add derived fields
        Pgrav_gas = -(dat.density*dat.gz_gas).where(dat.z>0).sum(dim='z')*dz
        Pgrav_starpar = -(dat.density*dat.gz_starpar).where(dat.z>0).sum(dim='z')*dz
        Pgrav_ext = -(dat.density*dat.gz_ext).where(dat.z>0).sum(dim='z')*dz

        dat = dat.drop(['gz_sg', 'gz_starpar', 'gz_gas', 'gz_ext'])
        add_derived_fields(dat, ['surf','H','sz'])
        dat = dat.drop('velocity3')

        dat['pressure'] = dat.pressure.sel(z=0, method='nearest')
        dat['Pturb'] = dat.Pturb.sel(z=0, method='nearest')
        dat['density'] = dat.density.sel(z=0, method='nearest')

        # delineate the ring by applying a mass cut
        surf_th, mask = mask_ring_by_mass(dat, mf_crit=args.mf_crit,
                Rmax=args.Rmax)
        area = _get_area(dat.where(mask))

        surf = dat.surf.where(mask).mean().values[()]
        msp = grid_msp(s, num, 0, 1e10)
        surfstar = msp.where(mask).sum().values[()]/area
        agebin = 1/s.u.Myr
        msp = grid_msp(s, num, 0, agebin)
        surfsfr = msp.where(mask).sum().values[()]/area/agebin

        n0 = dat.density.where(mask).mean().values[()]
        H = dat.H.where(mask).mean().values[()]
        Hs = np.sqrt((sp.mass*sp.x3**2).sum()/sp.mass.sum())
        sz = dat.sz.where(mask).mean().values[()]

        Pgrav_gas = Pgrav_gas.where(mask).mean().values[()]
        Pgrav_starpar = Pgrav_starpar.where(mask).mean().values[()]
        Pgrav_ext = Pgrav_ext.where(mask).mean().values[()]
        Pturb = dat.Pturb.where(mask).mean().values[()]
        Pth = dat.pressure.where(mask).mean().values[()]

        np.savetxt("{}/{}.{:04d}.txt".format(outdir,fname,num),
                [t, surf, surfstar, surfsfr, n0, H, Hs, sz,
                    Pgrav_gas, Pgrav_starpar, Pgrav_ext, Pturb, Pth, area])
