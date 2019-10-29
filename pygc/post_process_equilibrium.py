import numpy as np
import os
from pygc.ring import do_average

if __name__ == '__main__':
    import argparse
    from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC

    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('-v', '--verbosity', action='count',
                        help='increase output verbosity')
    parser.add_argument('--outdir', default=None, help='output directory')
    parser.add_argument('--mpi', action='store_true', help='enable mpi')
    parser.add_argument('--twophase', action='store_true')
    parser.add_argument('--pgravmask', action='store_true')
    args = parser.parse_args()

    if args.mpi:
        from mpi4py import MPI
        from pyathena.util.split_container import split_container
        COMM = MPI.COMM_WORLD
        myrank = COMM.rank
    else:
        myrank = 0

    if args.outdir==None:
        outdir=args.indir+'/postproc_equilibrium'
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
        print('[rank, mysteps]:', myrank, mynums)
    else:
        mynums=nums

    # load simulation and perform local time-average
    s = LoadSimTIGRESSGC(args.indir)

    for num in mynums:
        ds = do_average(s, num, twophase=args.twophase, pgravmask=args.pgravmask)
        np.savetxt("{}/ringavg.{:04d}.txt".format(outdir,num), ds)
