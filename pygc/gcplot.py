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

def draw_tigress_gc(indir, nums, all=None, projection=None, history=None):
    """
    =========================================================
    Description | serial function to generate figures
    Author      | Sanghyuk Moon
    =========================================================
    indir       | input simulation directory (indir/vtk; indir/starpar; etc.);
    nums        | selected snapshots [start, end]
    =========================================================
    """
    from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
    from pyathena.tigress_gc.plt_tigress_gc import plt_proj_density, plt_all, plt_history
    import time
    import matplotlib.pyplot as plt

    if all:
        fsize = (32,18)
    elif projection:
        fsize = (22,12)

    fig = plt.figure(figsize=fsize, dpi=60)

    # Measure execution time
    time0 = time.time()

    s = LoadSimTIGRESSGC(indir, verbose=False)
    
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
    print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
    print('################################################')
    print('')

    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    args = parser.parse_args()
    draw_tigress_gc(args.indir, np.arange(args.start,args.end+1),
                    all=True, projection=False)
