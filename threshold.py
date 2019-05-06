import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as c
from astropy import units as u
from scipy.optimize import bisect

def func(T,gm,dx):
    return T**1.5*np.exp(-92/T)-0.6*gm*dx**2

if __name__ == '__main__':
    Ta=10
    Tb=1e5
    gm=1
    muH=1.4271
    dx=np.logspace(-1,2,100)
    Tth=np.zeros(100)
    dth=np.zeros(100)
    for i in range(100):
        Tth[i] = bisect(func,Ta,Tb,args=(gm,dx[i]))
        dth[i] = ((8.86/np.pi*c.k_B*(Tth[i])*u.K/muH/c.m_p/c.G/((dx[i])*u.pc)**2)/(muH*c.m_p)).cgs.value
    plt.loglog(dx,Tth,'b-')
    plt.loglog(dx,dth,'g-')

    Ta=10
    Tb=1e5
    gm=1000
    muH=1.4271
    dx=np.logspace(-1,2,100)
    Tth=np.zeros(100)
    dth=np.zeros(100)
    for i in range(100):
        Tth[i] = bisect(func,Ta,Tb,args=(gm,dx[i]))
        dth[i] = ((8.86/np.pi*c.k_B*(Tth[i])*u.K/muH/c.m_p/c.G/((dx[i])*u.pc)**2)/(muH*c.m_p)).cgs.value
    plt.loglog(dx,Tth,'b--')
    plt.loglog(dx,dth,'g--')

    plt.xlim(1e-1,1e2)
    plt.ylim(1e0,1e7)
    plt.tight_layout()
    plt.savefig("th.pdf")
