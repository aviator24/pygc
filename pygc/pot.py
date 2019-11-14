from pyathena.util.units import Units
import numpy as np
from astropy import units as au
from astropy import constants as ac

u = Units()

class MHubble():
    """Modified Hubble potential"""
    def __init__(self, rb, rhob):
        self.r_b = rb
        self.rho_b = rhob
    
    def Menc(self, r):
        """enclosed mass within radius r [pc]
        return in Msun
        """
        M = 4.*np.pi*self.r_b**3*self.rho_b*(np.log(r/self.r_b
            + np.sqrt(1.+r**2/self.r_b**2))
            - r/self.r_b/np.sqrt(1.+r**2/self.r_b**2))
        return M
    
    def vcirc(self, R, z):
        """Circular velocity at radius (R,z) [pc]
        return in km/s
        """
        r = np.sqrt(R**2+z**2)
        vsq = 4*np.pi*ac.G.to('km2 s-2 pc Msun-1').value\
                *self.rho_b*self.r_b**2*(self.r_b/r*np.log(r/self.r_b
                +np.sqrt(1.+r**2/self.r_b**2))
                -1./np.sqrt(1.+r**2/self.r_b**2))
        return np.sqrt(vsq)*R/r

    def rho(self, r):
        """stellar density at radius r [pc]
        return in Msun/pc^3
        """
        return self.rho_b / (1.+r**2/self.r_b**2)**1.5

    def rho_eff(self, R, z):
        """Effective bulge stellar density at (R,z) [pc],
        defined as g_z(R,z) = -4 pi G rho_eff(R,z) z (OML10)
        return in Msun/pc^3
        """
        r = np.sqrt(R**2 + z**2)
        return self.rho_b*(self.r_b/r)**2*(self.r_b/r*np.log(r/self.r_b
            + np.sqrt(1.+r**2/self.r_b**2)) - 1./np.sqrt(1.+r**2/self.r_b**2))

    def gz(self, R, z):
        """Vertical gravitational acceleration at (R,z) [pc]
        return in km/s/Myr
        """
        gz = -4*np.pi*ac.G.to("km s^-1 Myr^-1 Msun^-1 pc^2").value\
                *self.rho_eff(R, z)*z
        return gz

    def gz_linear(self, R, z):
        """Linearly approximated vertical gravitational acceleration at (R,z) [pc]
        return in km/s/Myr
        """
        return -4*np.pi*ac.G.to("km s^-1 Myr^-1 Msun^-1 pc^2").value\
                *self.rho_eff(R,0)*z

def vcirc_KE17(R):
    """Kim & Elmegreen (2017) rotation curve (R is given in pc)
    return in km/s
    """
    return 215 + 95*np.tanh((R-70)/60) - 50*np.log10(R) + 1.5*(np.log10(R))**3


