import numpy as np
from pyathena.util.units import Units
from astropy import units as u
unit=Units()
iflw_d0=69.2125
iflw_v0=160
iflw_w=36
iflw_h=36
iflw_mu=0.96592583 # 15 degree
iflw_b=72
Mdot=2*(iflw_d0*unit.density*iflw_v0*unit.velocity*iflw_mu
        *iflw_w*unit.length*iflw_h*unit.length).to(u.M_sun/u.yr)
print(Mdot)
