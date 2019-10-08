from pyathena.classic.cooling import coolftn
from pygc.derived_fields import add_derived_fields
from pygc.pot import gz_ext
import xarray as xr
import pandas as pd
import numpy as np
import pickle

#@profile
def dataset_tavg(s, nums, Twarm=2.0e4, sum=False):
    """Do time-avearge on Datasets and return the averaged Dataset.

    Parameters
    ----------
    s : LoadSimTIGRESSGC
        LoadSimTIGRESSGC instance to be analyzed.
    nums : int
        list of snapshot numbers
    Twarm : float
        A demarcation temperature for two-phase medium
    """

    cf = coolftn()
    fields = ['density','velocity','pressure','gravitational_potential']

    # load a first vtk
    ds = s.load_vtk(num=nums[0])
    dat = ds.get_field(fields, as_xarray=True)
    add_derived_fields(dat, fields=['T','Pturb'], in_place=True)
    tmp = dat.where(dat.T < Twarm, other=1e-38) # two-phase medium
    dat = xr.concat([dat,tmp], pd.Index(['all', '2p'], name='phase'))
    dat = dat.drop('T')
    # gravitational field exists even when the matter does not exist.
    dat.gravitational_potential.loc[{'phase':'2p'}] = \
            dat.gravitational_potential.loc[{'phase':'all'}]
    add_derived_fields(dat, fields=['R','gz_sg','Pgrav'], in_place=True)

    # loop through vtks
    for num in nums[1:]:
        ds = s.load_vtk(num=num)
        tmp = ds.get_field(fields, as_xarray=True)
        add_derived_fields(tmp, fields=['T','Pturb'], in_place=True)
        tmp2 = tmp.where(tmp.T < Twarm, other=1e-38)
        tmp = xr.concat([tmp,tmp2], pd.Index(['all', '2p'], name='phase'))
        tmp = tmp.drop('T')
        # gravitational field exists even when the matter does not exist.
        tmp.gravitational_potential.loc[{'phase':'2p'}] = \
                tmp.gravitational_potential.loc[{'phase':'all'}]
        add_derived_fields(tmp, fields=['R','gz_sg','Pgrav'], in_place=True)

        # combine
        dat += tmp

    if not sum:
        dat /= len(nums)
    return dat

if __name__ == '__main__':
    import argparse
    from mpi4py import MPI
    from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
    from pyathena.util.split_container import split_container
    import pickle


    COMM = MPI.COMM_WORLD

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='selected model')
    parser.add_argument('start', type=int, help='start index')
    parser.add_argument('end', type=int, help='end index')
    parser.add_argument('-v', '--verbosity', action='count',
                        help='increase output verbosity')
    args = parser.parse_args()

    if COMM.rank == 0:
        if args.verbosity is not None:
            if args.verbosity >= 2:
                print("Running '{}'".format(__file__))
            if args.verbosity >= 1:
                print("selected model: {}".format(args.model))
                print("generating time average between snapshot {} to {}"
                        .format(args.start, args.end))
    nums = np.arange(args.start,args.end+1)
    if COMM.rank == 0:
        nums = split_container(nums, COMM.size)
    else:
        nums = None
    mynums = COMM.scatter(nums, root=0)
    print('[rank, mysteps]:', COMM.rank, mynums)

    # load simulation and perform local time-average
    s = LoadSimTIGRESSGC("/data/shmoon/TIGRESS-GC/{}".format(args.model))
    dat = dataset_tavg(s, mynums, sum=True)

    # dump local time-averages
    with open("{}.tavg.{}".format(args.model, COMM.rank), "wb") as handle:
        pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    COMM.Barrier()

    # combine local time-averages into global time-average dump
    if COMM.rank == 0:
        for i in range(1, COMM.size):
            dat += pickle.load(open("{}.tavg.{}".format(args.model, COMM.rank),
                "rb"))
        dat /= (args.end - args.start + 1)
        dat.attrs.update({'ts':s.load_vtk(num=args.start).domain['time']*s.u.Myr,
                          'te':s.load_vtk(num=args.end).domain['time']*s.u.Myr})
        with open("{}.tavg".format(args.model), "wb") as handle:
            pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
