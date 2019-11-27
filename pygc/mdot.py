#!/usr/bin/env python
"""
=====================================================================
Description | find inflow density that is consistent with target Mdot
Author      | Sanghyuk Moon
=====================================================================
"""
import numpy as np
import astropy.units as au
from pygc.pot import MHubble
from pyathena.util.units import Units
from scipy.optimize import bisect

target_mdot = 1.0

def Mdot(iflw_d0):
    Mdot = 0
    for x0 in np.arange(iflw_b+hdx, iflw_b+iflw_w, dx):
        iflw_v0 = (Lz0 - (x0**2+y0**2)*Omega_0)/(x0*iflw_mu
                - y0*np.sqrt(1.-iflw_mu**2))
        Mdot += iflw_d0*iflw_v0*iflw_mu
    Mdot *= dx*iflw_h
    Mdot *= u.Msun / u.Myr / 1e6
    return Mdot - target_mdot

if __name__ == '__main__':
    u=Units()
    m = MHubble(120, 265)
    Omega_0 = 0.04*u.Myr
    iflw_mu=np.cos(10*au.deg)
    Rring = 100
    iflw_b=80
    iflw_w=40
    iflw_h=40
    dx = 4
    hdx = dx>>1
    y0 = -256-hdx
    Lz0 = Rring*m.vcirc(Rring,0,0)
    iflw_d = bisect(Mdot, 1e-2, 1e3)
    print(iflw_d)
    print(Mdot(iflw_d))
