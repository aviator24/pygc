#!/usr/bin/env python
"""
=================================================
Description | main plotting script for TIGRESS-GC
Author      | Sanghyuk Moon
=================================================
"""
import argparse
import numpy as np
from mpi4py import MPI
from os import path as osp

def draw_tigress_gc(COMM, model, nums, all=None, projection=None, history=None):
    """
    =========================================================
    Description | MPI parallel function to generate figures
    Author      | Sanghyuk Moon
    =========================================================
    COMM        | MPI communicator
    model       | selected model;
                  default loc: /home/smoon/data/gc/your_model
    nums        | selected snapshots [start, end]
    =========================================================
    """
    from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
    from pyathena.tigress_gc.plt_tigress_gc import plt_proj_density, plt_all, plt_history
    from pyathena.util.split_container import split_container
    import time
    import matplotlib.pyplot as plt

    basename = "/data/shmoon/TIGRESS-GC/"
    if all:
        fsize = (32,18)
    elif projection:
        fsize = (22,12)

    fig = plt.figure(figsize=fsize, dpi=60)

    # Measure execution time
    time0 = time.time()

    s = LoadSimTIGRESSGC(basename+model, verbose=False)
    
    if COMM.rank == 0:
        nums = split_container(nums, COMM.size)
    else:
        nums = None
    
    mynums = COMM.scatter(nums, root=0)
    print('[rank, mysteps]:', COMM.rank, mynums)

    for num in mynums:
        dirname = osp.dirname(s.files['vtk'][0])
        fvtk = osp.join(dirname, '{0:s}.{1:04d}.vtk'.format(s.problem_id, num))
        if not osp.exists(fvtk):
            continue
        print(num, end=' ')
        if all:
            plt_all(s, num, fig)
        elif projection:
            plt_proj_density(s, num, fig)
        fig.clf()
    
    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Done with model', model)
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')

    plt.close(fig)

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='selected model')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('-v', '--verbosity', action='count',
                        help='increase output verbosity')
    parser.add_argument('-a', '--all', action='store_true',
                        help='draw everything in a single panel')
    parser.add_argument('-p', '--projection', action='store_true',
                        help='draw density projection')
    args = parser.parse_args()
    if COMM.rank == 0:
        if args.verbosity is not None:
            if args.verbosity >= 2:
                print("Running '{}'".format(__file__))
            if args.verbosity >= 1:
                print("selected model: {}".format(args.model))
                print("drawing from {} to {}".format(args.start, args.end))
    draw_tigress_gc(COMM, args.model, np.arange(args.start,args.end+1),
                    all=args.all, projection=args.projection)
