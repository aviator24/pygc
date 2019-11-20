from pyathena.util.units import Units
import numpy as np
from astropy import units as au
from astropy import constants as ac

class pot():
    """
    Base class for potential models
    unit system : [M] = Msun, [V] = km/s, [L] = pc
    """
    def __init__(self):
        self.G = ac.G.to('Msun-1 pc km2 s-2').value

class rigid(pot):
    """rigid body rotation"""
    def __init__(self, Omg):
        super().__init__()
        self.Omg = Omg
        self.Omg2 = Omg**2
        self.rho = 3*self.Omg2/(4*np.pi*self.G)
    def Menc(self, x, y, z):
        r3 = (x**2+y**2+z**2)**1.5
        return r3*self.Omg2/self.G
    def Phi(self, x, y, z):
        r2 = x**2 + y**2 + z**2
        return 0.5*r2*Omg2
    def gx(self, x, y, z):
        return -x*self.Omg2
    def gy(self, x, y, z):
        return -y*self.Omg2
    def gz(self, x, y, z):
        return -z*self.Omg2

class Ferrers(pot):
    """n=1 Ferrers bar"""
    # TODO 3D generalization
    def __init__(self, rhobar, a, b):
        super().__init__()
        self.rhobar = rhobar
        self.a = a
        self.b = b
        e = np.sqrt(1 - b**2/a**2)
        self.Mbar = 8.*np.pi*a*b**2*rhobar/15.
        self.W00 = 1./a/e*np.log((1.+e)/(1.-e))
        self.W10 = 2./a**3/e**2*(0.5/e*np.log((1.+e)/(1.-e)) - 1.)
        self.W01 = 1./a**3/e**2*(1./(1.-e**2)
                - 0.5*np.log((1.+e)/(1.-e))/e)
        self.W11 = (self.W01-self.W10)/(a**2*e**2)
        self.W20 = 2./3.*(1./a**5/(1.-e**2) - self.W11)
        self.W02 = 0.25*(2./a**5/(1.-e**2)**2 - self.W11)
    def rho(self, x, y):
        g2 = (y/self.a)**2 + (x/self.b)**2
        den = np.zeros(g2.shape)
        den[g2<1] = self.rhobar*(1.-g2[g2<1])
        return den
    def Phi(self, x, y):
        coeff = -0.5*np.pi*self.G*self.a*self.b**2*self.rhobar
        return coeff*(self.W00 - 2.*self.W01*x**2 - 2.*self.W10*y**2
                + self.W02*x**4 + 2.*self.W11*x**2*y**2 + self.W20*y**4)
    def gx(self, x, y):
        coeff = 0.5*np.pi*self.G*self.a*self.b**2*self.rhobar
        return coeff*(-4.*self.W01*x + 4.*self.W02*x**3 + 4.*self.W11*x*y**2)
    def gy(self, x, y):
        coeff = 0.5*np.pi*self.G*self.a*self.b**2*self.rhobar
        return coeff*(-4.*self.W10*y + 4.*self.W20*y**3 + 4.*self.W11*x**2*y)

class MHubble(pot):
    """Modified Hubble potential"""
    def __init__(self, rb, rhob):
        super().__init__()
        self.r_b = rb
        self.rho_b = rhob
    def Menc(self, x, y, z):
        r = np.sqrt(x**2+y**2+z**2)
        M = 4.*np.pi*self.r_b**3*self.rho_b*(np.log(r/self.r_b
            + np.sqrt(1.+r**2/self.r_b**2))
            - r/self.r_b/np.sqrt(1.+r**2/self.r_b**2))
        return M
    def vcirc(self, x, y, z):
        r = np.sqrt(x**2+y**2+z**2)
        return np.sqrt(self.G*self.Menc(x,y,z)/r)
    def rho(self, r):
        return self.rho_b / (1.+r**2/self.r_b**2)**1.5
    def gx(self, x, y, z):
        r = np.sqrt(x**2+y**2+z**2)
        return -self.G*self.Menc(x,y,z)*x/r**3
    def gy(self, x, y, z):
        r = np.sqrt(x**2+y**2+z**2)
        return -self.G*self.Menc(x,y,z)*y/r**3
    def gz(self, x, y, z):
        r = np.sqrt(x**2+y**2+z**2)
        return -self.G*self.Menc(x,y,z)*z/r**3
def vcirc_KE17(R):
    """Kim & Elmegreen (2017) rotation curve (R is given in pc)
    return in km/s
    """
    return 215 + 95*np.tanh((R-70)/60) - 50*np.log10(R) + 1.5*(np.log10(R))**3
