from pyathena.tigress_gc.load_sim_tigress_gc import LoadSimTIGRESSGC
from pyathena.util.units import Units
import pandas as pd
import numpy as np
import xarray as xr
from pyathena.classic.cooling import coolftn
from pygc.util import wmean
import pickle

Thot2, Thot1, Twarm, Tcold = 5.e5, 2.e4, 5050, 184.

def create_Rzprof(s, num, iphase):
    """
    Transform dataset from (x,y,z) to (R,z).
    Scale height is a function of R only.
    Write output at the current directory.
    """
    axis_idx = dict(x=0, y=1, z=2)
    # load vtk files
    ds = s.load_vtk(num=num)
    dat0 = ds.get_field(field=['density','velocity','pressure',
        'gravitational_potential'], as_xarray=True)

    # add additional fields
    cf = coolftn()
    u = Units()
    pok = dat0['pressure']*u.pok
    T1 = pok/(dat0['density']*u.muH)
    del pok
    dat0['temperature'] = xr.DataArray(cf.get_temp(T1.values),
            coords=T1.coords, dims=T1.dims)
    del T1
    rcyl = xr.DataArray(np.sqrt(dat0.x**2+dat0.y**2), name='rcyl')
    dat0['rcyl'] = xr.broadcast(rcyl, dat0)[0]
    cos, sin = dat0.x/dat0.rcyl, dat0.y/dat0.rcyl
    dat0['vr'] = dat0.velocity1*cos + dat0.velocity2*sin
    dat0['vt'] = -dat0.velocity1*sin + dat0.velocity2*cos
    del cos
    del sin
    dat0 = dat0.drop(['velocity1', 'velocity2'])
    dat0['vz'] = dat0.velocity3
    dat0 = dat0.drop('velocity3')

    # phase labels
    if iphase==0: # all
        phase = False
    elif iphase==1: # cold
        phase = dat0['temperature'] < Tcold
    elif iphase==2: # unstable
        phase = (dat0['temperature'] > Tcold)&(dat0['temperature'] < Twarm)
    elif iphase==3: # warm
        phase = (dat0['temperature'] > Twarm)&(dat0['temperature'] < Thot1)
    elif iphase==4: # hot1
        phase = (dat0['temperature'] > Thot1)&(dat0['temperature'] < Thot2)
    elif iphase==5: # hot2
        phase = dat0['temperature'] > Thot2
    elif iphase==6: # warm+unstable
        phase = (dat0['temperature'] > Tcold)&(dat0['temperature'] < Thot1)
    elif iphase==7: # warm+unstable+cold
        phase = dat0['temperature'] < Thot1
    elif iphase==8: # unstable+cold
        phase = dat0['temperature'] < Twarm
    else:
        raise Exception("invalid iphase")

    if iphase==0:
        dat = dat0
    else:
        dat = dat0.where(phase)

    # Make radial bins
    Rmin, Rmax, dR = 0, 256, 10
    Redges = np.arange(Rmin, Rmax, dR)
    Rbins = 0.5*(Redges[1:]+Redges[:-1])
    Nr = len(Redges)-1

    newdat = []
    for i in range(Nr):
        Rl, Rr = Redges[i], Redges[i+1]
        mask = (Rl < dat.rcyl)&(dat.rcyl < Rr)

        # calculate midplane turbulent pressure
        Pturb = (dat.density.interp(z=0).where(mask)*dat.vz.interp(z=0).where(mask)**2).mean()

        # calculate volume filling factor
        mask0 = (Rl < dat0.rcyl)&(dat0.rcyl < Rr)
        fvolume = ((~np.isnan(dat.rcyl.where(mask, drop=True))).sum(dim=['x','y'])
                / (~np.isnan(dat0.rcyl.where(mask0, drop=True))).sum(dim=['x','y']))

        # average over an annulus
        datRz = dat.where(mask).mean(dim=['x','y']).expand_dims(dim={'R':Rbins[i:i+1]})
        datRz['fvolume'] = fvolume.expand_dims('R')
        datRz['Pturb'] = Pturb.expand_dims('R')
        newdat.append(datRz)
    newdat = xr.merge(newdat)
    newdat = newdat.drop('rcyl')
    newdat = newdat.rename({'density':'rho', 'pressure':'Pth',
        'gravitational_potential':'Phi', 'temperature':'T'})
    fname = "{}.{:04d}.phase{}.Rzprof".format(s.problem_id, num, iphase)
    with open(fname, "wb") as handle:
        pickle.dump(newdat, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_sfr(sp, tbinMyr):
    """ SFR in Msun / yr """
    return sp[sp['mage'] < tbinMyr]['mass'].sum() / (tbinMyr*1e6)

def sfrprof(s, num):
    ds = s.load_vtk(num=num)
    time = ds.domain['time']*s.u.Myr
    sp = s.load_starpar_vtk(num=num)
    sp['mass'] = sp['mass']*s.u.Msun
    sp['age'] = sp['age']*s.u.Myr
    sp['mage'] = sp['mage']*s.u.Myr
    sfr10 = get_sfr(sp, 10) 

    # Make radial bins
    Rmin, Rmax, dR = 0, 256, 10
    Redges = np.arange(Rmin, Rmax, dR)
    Rbins = 0.5*(Redges[1:]+Redges[:-1])
    Nr = len(Redges)-1
    
    sfr10 = []
    for i in range(Nr):
        Rl, Rr = Redges[i], Redges[i+1]
        R = np.sqrt(sp.x1**2 + sp.x2**2)
        area = np.pi*(Rr**2-Rl**2)
        sfr10.append(get_sfr(sp[(Rl < R)&(R < Rr)], 10)/area)
    sfr10 = np.array(sfr10)*1e6 # sfr [ Msun / yr / kpc^2 ]
    return Rbins, sfr10
