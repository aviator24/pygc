#!/usr/bin/env python
"""
===========================================
Description | Return time-averaged snapshot 
Author      | Sanghyuk Moon
===========================================
"""

from pygc.util import sum_dataset
from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
import numpy as np
import pickle
import os

if __name__ == '__main__':
    import argparse

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
        outdir=args.indir+'/postproc_tavg'
    else:
        outdir=args.outdir
    if (~os.path.exists(outdir))&(myrank==0):
        os.mkdir(outdir)

    if args.twophase:
        if args.mpi:
            fname_local = "{}/gc.{:04d}.{:04d}.2p.pkl.{}".format(outdir, args.start, args.end, COMM.rank)
        fname_global = "{}/gc.{:04d}.{:04d}.2p.pkl".format(outdir, args.start, args.end)
    else:
        if args.mpi:
            fname_local = "{}/gc.{:04d}.{:04d}.pkl.{}".format(outdir, args.start, args.end, COMM.rank)
        fname_global = "{}/gc.{:04d}.{:04d}.pkl".format(outdir, args.start, args.end)

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
    dat = sum_dataset(s, mynums, twophase=args.twophase)

    if args.mpi:
        # dump local sum
        with open(fname_local, "wb") as handle:
            pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        COMM.Barrier()
    
        # combine local time-averages into global time-average dump
        if myrank == 0:
            for i in range(1, COMM.size):
                dat += pickle.load(open(fname_global+".{}".format(i), "rb"))
            dat /= (args.end - args.start + 1)
            dat.attrs.update({'ts':s.load_vtk(num=args.start).domain['time'],
                              'te':s.load_vtk(num=args.end).domain['time'],
                              'domain':s.domain})
            # dump global time-average
            with open(fname_global, "wb") as handle:
                pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        COMM.Barrier()
        os.remove(fname_local)
    else:
        dat /= (args.end - args.start + 1)
        dat.attrs.update({'ts':s.load_vtk(num=args.start).domain['time'],
                          'te':s.load_vtk(num=args.end).domain['time'],
                          'domain':s.domain})
        with open(fname_global, "wb") as handle:
            pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
