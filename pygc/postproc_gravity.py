#!/usr/bin/env python
"""
=================================================
Description | Run gravity post-processing
Author      | Sanghyuk Moon
=================================================
"""
import argparse
import numpy as np
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('--mpi', action='store_true', help='enable mpi')
    args = parser.parse_args()

    if args.mpi:
        from mpi4py import MPI
        from pyathena.util.split_container import split_container
        COMM = MPI.COMM_WORLD
        myrank = COMM.rank
    else:
        myrank = 0

    outdir=args.indir+'/postproc_gravity'
    if (~os.path.exists(outdir))&(myrank==0):
        raise FileNotFoundError('create directory "postproc_gravity"')

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

    # run athena post_processing_gravity problem
    for num in mynums:
        datafile=args.indir+'/vtk/gc.{:04d}.vtk'.format(num)
        starparfile=args.indir+'/starpar/gc.{:04d}.starpar.vtk'.format(num)
        cmd=['./athena', '-i', 'athinput.post_processing_gravity',
             'output1/num={:04d}'.format(num),
             'problem/datafile='+datafile,
             'problem/starparfile='+starparfile]
        os.chdir(outdir)
        p = subprocess.Popen(cmd,stdout=subprocess.PIPE)
        for line in p.stdout:
            print(line, end='')
            p.wait()
