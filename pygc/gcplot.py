#!/usr/bin/env python
"""
=================================================
Description | main plotting script for TIGRESS-GC
Author      | Sanghyuk Moon
=================================================
"""
from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
from pyathena.tigress_gc.plt_tigress_gc import plt_all
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time

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

    # Measure execution time
    time0 = time.time()

    # load simulation
    s = LoadSimTIGRESSGC(args.indir, verbose=False)

    # create figure instance
    fsize = (32,18)
    fig = plt.figure(figsize=fsize, dpi=60)

    for num in mynums:
        dirname = os.path.dirname(s.files['vtk_id0'][0])
        fvtk = os.path.join(dirname, '{0:s}.{1:04d}.vtk'.format(s.problem_id, num))
#        if not os.path.exists(fvtk):
#            dirname = os.path.dirname(s.files['vtk'][0])
#            fvtk = os.path.join(dirname, '{0:s}.{1:04d}.vtk'.format(s.problem_id, num))
        if not os.path.exists(fvtk):
            continue
        print(num, end=' ')
        plt_all(s, num, fig, with_starpar=True)
        fig.clf()

    if args.mpi: 
        COMM.barrier()
    if myrank==0:
        print('')
        print('################################################')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')

    plt.close(fig)
