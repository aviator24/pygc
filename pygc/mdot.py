#!/usr/bin/env python
"""Computes iflw_d0 at given mass inflow rate.

  Typical usage example:
  ./mdot.py 1 1024 256
"""
import argparse
import numpy as np
from scipy.optimize import bisect
import astropy.units as au
from pygc.pot import MHubble, Log, Plummer
from pyathena.util.units import Units
u = Units()


def Mdot(iflw_d0):
    """
    Computes mass inflow rate.

    Calculate the mass inflow rate Mdot = \int rho v dx dy through the nozzle
    boundary, multiplied by a factor of two assuming there are two nozzles.

    Args:
        iflw_d0: the hydrogen number density inside the nozzle in units of [cm^-3]

    Returns:
        Total mass inflow rate through the two nozzles in units of [Msun/yr]
    """
    Mdot = 0
    for x0 in np.arange(iflw_b+hdx, iflw_b+iflw_w, dx):
        iflw_v0 = (Lz0 - (x0**2+y0**2)*Omega_0)/(x0*iflw_mu
                - y0*np.sqrt(1.-iflw_mu**2))
        Mdot += iflw_d0*iflw_v0*iflw_mu
    Mdot *= dx*iflw_h
    Mdot *= 2
    Mdot *= u.Msun / u.Myr / 1e6
    return Mdot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('Mdot', default=1, type=float,
            help='mass inflow rate')

    parser.add_argument('Lx', default=1024, type=float,
            help='domain size [-Lx, Lx]')

    parser.add_argument('Nx', default=256, type=float,
            help='number of cells')

    parser.add_argument('--Rring', default=500, type=float,
            help='ring radius             (default = 500)')

    parser.add_argument('--iflw_b', default=448, type=float,
            help='inflow impact parameter (default = 448)')

    parser.add_argument('--iflw_w', default=192, type=float,
            help='nozzle width            (default = 192)')

    parser.add_argument('--iflw_h', default=192, type=float,
            help='nozzle height           (default = 192)')

    parser.add_argument('--iflw_mu', default=10, type=float,
            help='nozzle angle            (default = 10 deg)')

    parser.add_argument('--rhob', default=50, type=float,
            help='bulge central density   (default = 50 Msun/pc^3)')

    parser.add_argument('--rb', default=250, type=float,
            help='bulge radius            (default = 250 pc)')

    parser.add_argument('--Mc', default=1.4e8, type=float,
            help='black hole mass         (default = 1.4e8 Msun)')

    parser.add_argument('--Rc', default=20, type=float,
            help='black hole radius       (default = 20 pc)')

    args = parser.parse_args()

    Omega_0 = 0.036
    dx = 2*args.Lx/args.Nx
    hdx = dx/2
    target_mdot = args.Mdot
    Rring = args.Rring
    iflw_b = args.iflw_b
    iflw_w = args.iflw_w
    iflw_h = args.iflw_h
    iflw_mu=np.cos(args.iflw_mu*au.deg)
    y0 = -args.Lx-hdx

    bul = MHubble(args.rb, args.rhob)
    BH = Plummer(args.Mc, args.Rc)

    vc = np.sqrt(bul.vcirc(Rring,0,0)**2 + BH.vcirc(Rring,0,0)**2)
    Lz0 = Rring*vc
    iflw_d = bisect(lambda x: Mdot(x)-target_mdot, 1e-2, 1e3)
    print("\nMdot = {}".format(Mdot(iflw_d) + target_mdot))
    print("iflw_d0 = {:.6f}".format(iflw_d))
