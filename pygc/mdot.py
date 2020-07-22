#!/usr/bin/env python
"""
=====================================================================
Description | find inflow density that is consistent with target Mdot
Author      | Sanghyuk Moon
=====================================================================
"""
import numpy as np
import astropy.units as au
from pygc.pot import MHubble, Log, Plummer
from pyathena.util.units import Units
from scipy.optimize import bisect

target_mdot = 2

def Mdot(iflw_d0):
    Mdot = 0
    for x0 in np.arange(iflw_b+hdx, iflw_b+iflw_w, dx):
        iflw_v0 = (Lz0 - (x0**2+y0**2)*Omega_0)/(x0*iflw_mu
                - y0*np.sqrt(1.-iflw_mu**2))
        Mdot += iflw_d0*iflw_v0*iflw_mu
    Mdot *= dx*iflw_h
    Mdot *= 2
    Mdot *= u.Msun / u.Myr / 1e6
    return Mdot - target_mdot

if __name__ == '__main__':
    u=Units()
    dx = 8
    hdx = dx>>1
    iflw_mu=np.cos(10*au.deg)
    Omega_0 = 0.036

###### Small ring #####
#    Rring =150
#    iflw_b=80
#    iflw_w=50
#    iflw_h=50
#    y0 = -256-hdx
#    bul = MHubble(250, 50)
#    BH = Plummer(1.4e8, 20)
###### Large ring #####
    Rring =600
    iflw_b=320
    iflw_w=200
    iflw_h=200
    y0 = -1024-hdx
    bul = MHubble(250, 50)
    BH = Plummer(1.4e8, 20)

    vc = np.sqrt(bul.vcirc(Rring,0,0)**2 + BH.vcirc(Rring,0,0)**2)
    Lz0 = Rring*vc
    iflw_d = bisect(Mdot, 1e-2, 1e3)
    print("{:.6f}".format(iflw_d))
    print(Mdot(iflw_d)+target_mdot)
