#!/usr/bin/env python
"""
=======================================================================
Description | Read snapshot, delineate ring, compute various quantities
Author      | Sanghyuk Moon
=======================================================================
"""
from pygc.util import add_derived_fields
from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
import argparse
import numpy as np
import os

Twarm = 2.e4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
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
        outdir=args.indir+'/postproc_warmcold'
    else:
        outdir=args.outdir
    if (~os.path.exists(outdir))&(myrank==0):
        os.mkdir(outdir)

    fname = 'gc'

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
        t = ds.domain['time']
        dat = ds.get_field(['density','velocity','pressure'])
        dat = dat.drop(['velocity1','velocity2'])
        add_derived_fields(dat, 'T')
        # select two-phase gas
        dat = dat.where(dat.T < Twarm)
        add_derived_fields(dat, ['sz','cs','H'])

        np.savetxt("{}/{}.{:04d}.txt".format(outdir,fname,num),
            [t, dat.sz, dat.cs, dat.H])
