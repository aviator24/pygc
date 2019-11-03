#!/usr/bin/env python
"""
========================================================
Description | Read snapshot and return Pth, Pturb, Pgrav
Author      | Sanghyuk Moon
========================================================
"""
from pygc.util import add_derived_fields
from pygc.pot import gz_ext
from pygc.ring import _get_area
from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
from pyathena.io.read_vtk import read_vtk
import argparse
import numpy as np
import os

Twarm=2.e4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
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
        outdir=args.indir+'/postproc_equil'
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
    dx = s.domain['dx'][0]
    dy = s.domain['dx'][1]
    dz = s.domain['dx'][2]

    for num in mynums:
        ds = s.load_vtk(num)
        t = ds.domain['time']
        dat = ds.get_field(['density', 'velocity', 'pressure',
            'gravitational_potential'], as_xarray=True)
        dat = dat.drop(['velocity1','velocity2'])
        if args.twophase:
            add_derived_fields(dat, ['T','gz_sg'])
            gz_sg = dat.gz_sg
            dat = dat.where((dat.T<Twarm))
            dat = dat.drop(['T'])
            dat['gz_sg'] = gz_sg
        else:
            add_derived_fields(dat, ['gz_sg'])
        dat = dat.drop(['gravitational_potential'])
    
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
    
        # calculate weights
        dat['Pgrav_gas'] = -(dat.density*dat.gz_gas).where(dat.z>0).sum(dim='z')*dz
        dat['Pgrav_starpar'] = -(dat.density*dat.gz_starpar).where(dat.z>0).sum(dim='z')*dz
        dat['Pgrav_ext'] = -(dat.density*dat.gz_ext).where(dat.z>0).sum(dim='z')*dz
        dat = dat.drop(['gz_sg', 'gz_starpar', 'gz_gas', 'gz_ext'])
    
        dat['density'] = dat.density.sel(z=0, method='nearest')
        dat['velocity3'] = dat.velocity3.sel(z=0, method='nearest')
        dat['pressure'] = dat.pressure.sel(z=0, method='nearest')
        add_derived_fields(dat, ['Pturb'])
        dat = dat.drop(['density','velocity3','z'])
        dat['Pgrav_gas'] = dat.Pgrav_gas.where(dat.Pturb>0)
        dat['Pgrav_starpar'] = dat.Pgrav_starpar.where(dat.Pturb>0)
        dat['Pgrav_ext'] = dat.Pgrav_ext.where(dat.Pturb>0)
        area = _get_area(dat)
        Pth = (dat.pressure*dx*dy).sum().values[()]/area
        Pturb = (dat.Pturb*dx*dy).sum().values[()]/area
        Pgrav_gas = (dat.Pgrav_gas*dx*dy).sum().values[()]/area
        Pgrav_starpar = (dat.Pgrav_starpar*dx*dy).sum().values[()]/area
        Pgrav_ext = (dat.Pgrav_ext*dx*dy).sum().values[()]/area

        np.savetxt("{}/{}.{:04d}.txt".format(outdir,fname,num),
                [t,Pth,Pturb,Pgrav_gas,Pgrav_starpar,Pgrav_ext,area])
