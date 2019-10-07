#!/usr/bin/env python
"""
================================================================
Description | convenience script for estimating mass inflow rate
Author      | Sanghyuk Moon
================================================================
"""
import numpy as np
from pyathena.util.units import Units

if __name__ == '__main__':
    u=Units()
    iflw_d0=92.3
    iflw_v0=120
    iflw_w=36
    iflw_h=36
    iflw_mu=0.96592583 # 15 degree
    iflw_b=72
    Mdot=2*(iflw_d0*u.density*iflw_v0*u.velocity*iflw_mu
            *iflw_w*u.length*iflw_h*u.length).to("Msun / yr")
    print(Mdot)
