#!/usr/bin/env python
"""
=======================================================================
Description | Read snapshot, delineate ring, compute various quantities
Author      | Sanghyuk Moon
=======================================================================
"""
import argparse
import numpy as np
import os
from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
from pygc.derived_fields import add_derived_fields
from pygc.ring import mask_ring_by_mass, grid_msp, _get_area

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

    for num in mynums:
        ds = s.load_vtk(num)
        dat = ds.get_field(['density','velocity','pressure'],
                as_xarray=True)
        dat = dat.drop(['velocity1','velocity2'])
        add_derived_fields(dat, 'T', in_place=True)
        # select two-phase gas
        dat = dat.where(dat.T < Twarm)
        add_derived_fields(dat, ['surf','Pturb'], in_place=True)
        dat['Pturb'] = dat.Pturb.sel(z=0, method='nearest')
        dat['density'] = dat.density.sel(z=0, method='nearest')
        # delineate the ring by applying a mass cut
        surf_th, mask = mask_ring_by_mass(dat, mf_crit=args.mf_crit,
                Rmax=args.Rmax)
        surf = dat.surf.where(mask).mean().values[()]
        Pturb = dat.Pturb.where(mask).mean().values[()]
        n0 = dat.density.where(mask).mean().values[()]
        agebin = 1/s.u.Myr
        msp = grid_msp(s, num, 0, agebin)
        sfrsurf = msp.where(mask).sum().values[()]/\
                _get_area(dat.where(mask))/agebin
        np.savetxt("{}/gc.{:04d}.txt".format(outdir,num),
                [surf, Pturb, sfrsurf, n0])
