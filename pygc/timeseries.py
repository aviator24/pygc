from pygc.derived_fields import add_derived_fields
import numpy as np
import os

def get_area(dm):
    return ((dm.Pturb > 0).sum() / (dm.domain['Nx'][0]*dm.domain['Nx'][1]) * (dm.domain['Lx'][0]*dm.domain['Lx'][1])).values[()] 

def timeseries(s, num, Tmax=2e4, Rmax=200):
    ds = s.load_vtk(num)
    dx = ds.domain['dx'][0]
    dy = ds.domain['dx'][1]
    dat = ds.get_field(['density','velocity','pressure',
        'gravitational_potential'], as_xarray=True)
    dat = dat.drop(['velocity1','velocity2'])
    add_derived_fields(dat, ['R','T','gz_sg'], in_place=True)
    dat = dat.drop(['gravitational_potential'])
    gz_sg = dat.gz_sg
    dat = dat.where((dat.T<Tmax)&(dat.R<Rmax))
    dat = dat.drop(['T'])
    dat['gz_sg'] = gz_sg
    add_derived_fields(dat, ['Pturb','Pgrav'], in_place=True)
    dat = dat.drop(['density','velocity3','gz_sg'])
    dat = dat.sel(z=1)
    area = get_area(dat)
    t = ds.domain['time']*s.u.Myr
    Pth = (dat.pressure*dx*dy).sum().values[()]*s.u.pok/area
    Pturb = (dat.Pturb*dx*dy).sum().values[()]*s.u.pok/area
    Pgrav = (dat.Pgrav*dx*dy).sum().values[()]*s.u.pok/area
    return [t,Pth,Pturb,Pgrav,area]

if __name__ == '__main__':
    import argparse
    from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC

    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='input simulation directory')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('Tmax', type=float)
    parser.add_argument('Rmax', type=float)
    parser.add_argument('-v', '--verbosity', action='count',
                        help='increase output verbosity')
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
        outdir=args.indir+'/timeseries'
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
        ds = timeseries(s, num, Tmax=args.Tmax, Rmax=args.Rmax)
        np.savetxt("{}/ts.{:.0f}.{:04d}.txt".format(outdir,args.Rmax,num), ds)
