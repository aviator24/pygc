import numpy as np
from astropy import units as au
from astropy import constants as ac

def Mbul(r, r_b=120, rho_b=265):
    """Bulge mass within radius r [pc]
    return in Msun
    """
    return 4.*np.pi*r_b**3*rho_b*(np.log(r/r_b + np.sqrt(1.+r**2/r_b**2))
            - r/r_b/np.sqrt(1.+r**2/r_b**2))

def vcirc(r, r_b=120, rho_b=265):
    """Circular velocity at radius r [pc]
    return in km/s
    """
    vsq = 4*np.pi*ac.G.to('km^2 s^-2 pc Msun^-1').value*rho_b*r_b**2*(
            r_b/r*np.log(r/r_b + np.sqrt(1.+r**2/r_b**2))
            - 1./np.sqrt(1.+r**2/r_b**2))
    return np.sqrt(vsq)

def vcirc_KE17(R):
    """Kim & Elmegreen (2017) rotation curve (R is given in pc)
    return in km/s
    """
    return 215 + 95*np.tanh((R-70)/60) - 50*np.log10(R) + 1.5*(np.log10(R))**3

def rhobul(r, r_b=120, rho_b=265):
    """Bulge stellar density at radius r [pc]
    return in Msun/pc^3
    """
    return rho_b / (1.+r**2/r_b**2)**1.5

def rhobul_eff(R, z, r_b=120, rho_b=265):
    """Effective bulge stellar density at (R,z) [pc],
    defined as g_z(R,z) = -4 pi G rho_eff(R,z) z (OML10)
    return in Msun/pc^3
    """
    r = np.sqrt(R**2 + z**2)
    return rho_b*(r_b/r)**2*(r_b/r*np.log(r/r_b + np.sqrt(1.+r**2/r_b**2))
            - 1./np.sqrt(1.+r**2/r_b**2))

def gz_ext(R, z, r_b=120, rho_b=265):
    """Vertical gravitational acceleration at (R,z) [pc]
    return in km/s/Myr
    """
    return -4*np.pi*ac.G.to("km s^-1 Myr^-1 Msun^-1 pc^2").value\
            *rhobul_eff(R,z,r_b=r_b,rho_b=rho_b)*z

def gz_ext_linear(R, z, r_b=120, rho_b=265):
    """Linearly approximated vertical gravitational acceleration at (R,z) [pc]
    return in km/s/Myr
    """
    return -4*np.pi*ac.G.to("km s^-1 Myr^-1 Msun^-1 pc^2").value\
            *rhobul_eff(R,0,r_b=r_b,rho_b=rho_b)*z
