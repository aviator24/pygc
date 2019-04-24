import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as c
import astropy.units as u
import sys
sys.path.insert(0,'/home/smoon/Dropbox/gc/Athena-TIGRESS/python/')
import pyathena as pa
def read2d(dir, prob, i, units=True):
    """ read vtk and return x, y, z, density, velocity """
    da = pa.AthenaDataSet(dir+"/id0/"+prob+".{:04g}".format(i)+".vtk")
    xmin=da.domain['left_edge']
    xmax=da.domain['right_edge']
    dx=da.domain['dx']
    x=np.arange(xmin[0],xmax[0],dx[0])+0.5*dx[0]
    y=np.arange(xmin[1],xmax[1],dx[1])+0.5*dx[1]
    z=np.arange(xmin[2],xmax[2],dx[2])+0.5*dx[2]
    den = da.read_all_data('density')
    sigma = (den*dx[2]).sum(axis=0)
    vel = da.read_all_data('velocity')
    T = da.read_all_data('temperature')
    cs = da.read_all_data('sound_speed')
    Lz = x[None,None,:]*vel[:,:,:,1] - y[None,:,None]*vel[:,:,:,0]
    dV = dx[0]*dx[1]*dx[2]
    Mtot = (den*dV).sum()
    if (units==False):
        x*=u.pc; y*=u.pc; z*=u.pc; dx*=u.pc;
        den*=(1.4271*c.m_p/u.cm**3); vel*=u.km/u.s; Lz*=u.pc*u.km/u.s;
        Mtot*=(0.0352571473967*u.M_sun)
    dic = {"x":x, "y":y, "z":z, "dx":dx[0], "dy":dx[1], "dz":dx[2],
            "den":den, "vel":vel, "Lz":Lz, "Mtot":Mtot, "T":T, "cs":cs, "Sig":sigma}
    return dic
