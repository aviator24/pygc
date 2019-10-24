#!/usr/bin/env python
"""
=================================================
Description | main plotting script for TIGRESS-GC
Author      | Sanghyuk Moon
=================================================
"""
import argparse
import numpy as np
from os import path as osp

def draw_tigress_gc(model, nums, all=None, projection=None, history=None, prefix=None):
    """
    =========================================================
    Description | serial function to generate figures
    Author      | Sanghyuk Moon
    =========================================================
    model       | selected model;
    nums        | selected snapshots [start, end]
    =========================================================
    """
    from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
    from pyathena.tigress_gc.plt_tigress_gc import plt_proj_density, plt_all, plt_history
    import time
    import matplotlib.pyplot as plt

    basename = prefix+"/"
    if all:
        fsize = (32,18)
    elif projection:
        fsize = (22,12)

    fig = plt.figure(figsize=fsize, dpi=60)

    # Measure execution time
    time0 = time.time()

    s = LoadSimTIGRESSGC(basename+model, verbose=False)
    
    for num in nums:
        dirname = osp.dirname(s.files['vtk'][0])
        fvtk = osp.join(dirname, '{0:s}.{1:04d}.vtk'.format(s.problem_id, num))
        if not osp.exists(fvtk):
            continue
        print(num, end=' ')
        if all:
            plt_all(s, num, fig, with_starpar=True)
        elif projection:
            plt_proj_density(s, num, fig)
        fig.clf()
    
    print('')
    print('################################################')
    print('# Done with model', model)
    print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
    print('################################################')
    print('')

    plt.close(fig)

if __name__ == '__main__':
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
    parser.add_argument('--prefix', default="/data/shmoon/TIGRESS-GC",
                        help='base directory for simulation data')
    args = parser.parse_args()
    if args.verbosity is not None:
        if args.verbosity >= 2:
            print("Running '{}'".format(__file__))
        if args.verbosity >= 1:
            print("selected model: {}".format(args.model))
            print("drawing from {} to {}".format(args.start, args.end))
    draw_tigress_gc(args.model, np.arange(args.start,args.end+1),
                    all=args.all, projection=args.projection, prefix=args.prefix)
