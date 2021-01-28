#!/usr/bin/env python
"""
=======================================================================
Description | Read snapshot, delineate ring, compute various quantities
Author      | Sanghyuk Moon
=======================================================================
"""

from pygc.util import add_derived_fields
from pygc.ring import mask_ring_by_mass, ring_avg
import pyathena as pa
import pickle
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
    parser.add_argument('--mf_crit', default=0.9, type=float, help='mass cut')
    parser.add_argument('--Rmax', type=float, help='radius cut')
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
    s = pa.LoadSim(args.indir)
    dat_tavg = pickle.load(open(args.indir+'/postproc_tavg/tavg.pkl','rb'))
    add_derived_fields(dat_tavg, 'surf')
    surf_th, mask = mask_ring_by_mass(dat_tavg, mf_crit=args.mf_crit, Rmax=args.Rmax)

    for num in mynums:
        ds = ring_avg(s, num, mask)
        np.savetxt("{}/{}.{:04d}.txt".format(outdir,fname,num), ds)
