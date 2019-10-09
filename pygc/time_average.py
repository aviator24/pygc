from pyathena.classic.cooling import coolftn
from pygc.derived_fields import add_derived_fields
from pygc.pot import gz_ext
import xarray as xr
import pandas as pd
import numpy as np
import pickle
import os

Twarm = 2.0e4
cf = coolftn()
def dataset_tavg(s, nums, twophase=False):
    """Do time-summation on Datasets and return the summed Dataset.

    Parameters
    ----------
    s : LoadSimTIGRESSGC instance to be analyzed.
    nums :list of snapshot numbers to add
    twophase : include two-phase gas only
    """

    fields = ['density','velocity','pressure','gravitational_potential']

    # load a first vtk
    ds = s.load_vtk(num=nums[0])
    dat = ds.get_field(fields, as_xarray=True)
    if twophase:
        Phi = dat.gravitational_potential
        add_derived_fields(dat, fields='T', in_place=True)
        dat = dat.where(dat.T < Twarm, other=0)
        dat = dat.drop('T')
        dat['gravitational_potential'] = Phi
    add_derived_fields(dat, fields=['R','Pturb','Pgrav'], in_place=True)

    # loop through vtks
    for num in nums[1:]:
        ds = s.load_vtk(num=num)
        tmp = ds.get_field(fields, as_xarray=True)
        if twophase:
            Phi = tmp.gravitational_potential
            add_derived_fields(tmp, fields='T', in_place=True)
            tmp = tmp.where(tmp.T < Twarm, other=0)
            tmp = tmp.drop('T')
            tmp['gravitational_potential'] = Phi
        add_derived_fields(tmp, fields=['R','Pturb','Pgrav'], in_place=True)
        # add
        dat += tmp
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
    parser.add_argument('--twophase', action='store_true',
                        help='include two-phase gas only')
    args = parser.parse_args()

    if args.twophase:
        fname_local = "{}.tavg.2p.{}".format(args.model, COMM.rank)
        fname_global = "{}.tavg.2p".format(args.model)
    else:
        fname_local = "{}.tavg.{}".format(args.model, COMM.rank)
        fname_global = "{}.tavg".format(args.model)

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
    dat = dataset_tavg(s, mynums, twophase=args.twophase)

    # dump local sum
    with open(fname_local, "wb") as handle:
        pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    COMM.Barrier()

    # combine local time-averages into global time-average dump
    if COMM.rank == 0:
        for i in range(1, COMM.size):
            dat += pickle.load(open(fname_global+".{}".format(i), "rb"))
        dat /= (args.end - args.start + 1)
        dat.attrs.update({'ts':s.load_vtk(num=args.start).domain['time']*s.u.Myr,
                          'te':s.load_vtk(num=args.end).domain['time']*s.u.Myr,
                          'domain':s.domain})
        # dump global time-average
        with open(fname_global, "wb") as handle:
            pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.remove(fname_local)
