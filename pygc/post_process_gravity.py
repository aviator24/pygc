#!/usr/bin/env python
"""
=================================================
Description | Run gravity post-processing
Author      | Sanghyuk Moon
=================================================
"""
import argparse
import numpy as np
from mpi4py import MPI
from os import path as osp
import subprocess
from pyathena.util.split_container import split_container

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='selected model')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('--prefix', default="/data/shmoon/TIGRESS-GC",
                        help='base directory for simulation data')
    args = parser.parse_args()

    nums = np.arange(args.start,args.end+1)
    if COMM.rank == 0:
        nums = split_container(nums, COMM.size)
    else:
        nums = None
    mynums = COMM.scatter(nums, root=0)
    print('[rank, mysteps]:', COMM.rank, mynums)


    for num in mynums:
        datafile=args.prefix+'/'+args.model+'/vtk/gc.{:04d}.vtk'.format(num)
        starparfile=args.prefix+'/'+args.model+'/starpar/gc.{:04d}.starpar.vtk'.format(num)
        cmd=['./athena', '-i', 'athinput.post_processing_gravity',
             'output1/num={:04d}'.format(num),
             'problem/datafile='+datafile,
             'problem/starparfile='+starparfile]

        p = subprocess.Popen(cmd,stdout=subprocess.PIPE)
        for line in p.stdout:
            print(line, end='')
            p.wait()
