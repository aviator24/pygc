"""
star particle integrator module
written by Sanghyuk Moon, March 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def shearacc(x,y,Omega_0=1.0,qshear=1.0):
    gx = 2*qshear*Omega_0**2*x - Omega_0**2*x
    gy = -Omega_0**2*y
    return gx, gy

def intsp(x, y, vx, vy, accel, N=1000, dt=1e-2, Omega_0=1.0, method="Boris"):
    """
    Star particle integrator
    x,y,vx,vy : initial conditions
    accel : functions that return the gravitational acceleration.
        gx, gy = accel(x,y)
    if method=Quinn, this should be qshear
    N : Total # of steps
    Omega_0 : angular frequency of the rotating reference frame
    method = ["leapfrog", "Quinn", "Boris"]; default : Boris
    """
    t = 0
    hdt = 0.5*dt
    tout = np.zeros(N)
    xout = np.zeros(N)
    yout = np.zeros(N)
    vxout = np.zeros(N)
    vyout = np.zeros(N)
    tout[0] = t
    xout[0] = x
    yout[0] = y
    vxout[0] = vx
    vyout[0] = vy
    if (method=="leapfrog"):
        for i in range(1,N):
            # kick
            gx,gy = accel(x,y)
            gx += Omega_0**2*x + 2*Omega_0*vy
            gy += Omega_0**2*y - 2*Omega_0*vx
            vx += hdt*gx
            vy += hdt*gy
            # drift
            x += vx*dt
            y += vy*dt
            # kick
            gx,gy = accel(x,y)
            gx += Omega_0**2*x + 2*Omega_0*vy
            gy += Omega_0**2*y - 2*Omega_0*vx
            vx += hdt*gx
            vy += hdt*gy
            # advance time
            t += dt
            tout[i] = t
            xout[i] = x
            yout[i] = y
            vxout[i] = vx
            vyout[i] = vy
    if (method=="Quinn"):
        qshear = accel
        omdt = Omega_0*dt
        for i in range(1,N):
            # kick
            Py = vy + 2*Omega_0*x
            vx += omdt*(Py - 2*Omega_0*x) + qshear*omdt*(Omega_0*x)
            vy -= omdt*vx
            # drift
            x += vx*dt
            y += vy*dt
            # kick
            vy -= omdt*vx
            Py = vy + 2*Omega_0*x
            vx += omdt*(Py - (2.0 - qshear)*Omega_0*x)
            # advance time
            t += dt
            tout[i] = t
            xout[i] = x
            yout[i] = y
            vxout[i] = vx
            vyout[i] = vy
    if (method=="Boris"):
        for i in range(1,N):
            # kick
            gx,gy = accel(x,y)
            gx += Omega_0**2*x
            gy += Omega_0**2*y
            vx += gx*0.5*hdt
            vy += gy*0.5*hdt
            tan = Omega_0*hdt
            cos = (1-tan**2)/(1+tan**2)
            sin = 2*tan/(1+tan**2)
            vpx = cos*vx + sin*vy
            vpy = -sin*vx + cos*vy
            vx = vpx + gx*0.5*hdt
            vy = vpy + gy*0.5*hdt
            # drift
            x += vx*dt
            y += vy*dt
            # kick
            gx,gy = accel(x,y)
            gx += Omega_0**2*x
            gy += Omega_0**2*y
            vx += gx*0.5*hdt
            vy += gy*0.5*hdt
            tan = Omega_0*hdt
            cos = (1-tan**2)/(1+tan**2)
            sin = 2*tan/(1+tan**2)
            vpx = cos*vx + sin*vy
            vpy = -sin*vx + cos*vy       
            vx = vpx + gx*0.5*hdt
            vy = vpy + gy*0.5*hdt
            # advance time
            t += dt
            tout[i] = t
            xout[i] = x
            yout[i] = y
            vxout[i] = vx
            vyout[i] = vy
    return tout,xout,yout,vxout,vyout

if __name__ == '__main__':
    amp = 0.4
    Omega_0 = 1.0
    qshear = 1.0
    kap = np.sqrt(2.0*(2.0-qshear))*Omega_0
    x = 0
    y = 2*amp/kap
    vx = amp*kap
    vy = 0

    dt = 1e-1
    t,x1,y1,vx1,vy1 = intsp(x,y,vx,vy,qshear,dt=dt,N=400,method="Quinn")
    t,x2,y2,vx2,vy2 = intsp(x,y,vx,vy,shearacc,dt=dt,N=400,method="Boris")
    x0 = amp*np.sin(kap*t)
    y0 = 2.0*amp*Omega_0/kap*np.cos(kap*t)
    vx0 = amp*kap*np.cos(kap*t)
    vy0 = -2.0*amp*Omega_0*np.sin(kap*t)
    ds1 = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    ds2 = np.sqrt((x2-x0)**2 + (y2-y0)**2)
    err11 = ds1.mean()
    err21 = ds2.mean()

    dt = 1e-2
    t,x1,y1,vx1,vy1 = intsp(x,y,vx,vy,qshear,dt=dt,N=4000,method="Quinn")
    t,x2,y2,vx2,vy2 = intsp(x,y,vx,vy,shearacc,dt=dt,N=4000,method="Boris")
    x0 = amp*np.sin(kap*t)
    y0 = 2.0*amp*Omega_0/kap*np.cos(kap*t)
    vx0 = amp*kap*np.cos(kap*t)
    vy0 = -2.0*amp*Omega_0*np.sin(kap*t)
    ds1 = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    ds2 = np.sqrt((x2-x0)**2 + (y2-y0)**2)
    err12 = ds1.mean()
    err22 = ds2.mean()

    dt = 1e-3
    t,x1,y1,vx1,vy1 = intsp(x,y,vx,vy,qshear,dt=dt,N=40000,method="Quinn")
    t,x2,y2,vx2,vy2 = intsp(x,y,vx,vy,shearacc,dt=dt,N=40000,method="Boris")
    
    x0 = amp*np.sin(kap*t)
    y0 = 2.0*amp*Omega_0/kap*np.cos(kap*t)
    vx0 = amp*kap*np.cos(kap*t)
    vy0 = -2.0*amp*Omega_0*np.sin(kap*t)
    ds1 = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    ds2 = np.sqrt((x2-x0)**2 + (y2-y0)**2)
    err13 = ds1.mean()
    err23 = ds2.mean()
    E1 = 0.5*(vx1**2 + vy1**2) - qshear*(Omega_0*x1)**2
    E2 = 0.5*(vx2**2 + vy2**2) - qshear*(Omega_0*x2)**2
    fig, ax = plt.subplots(1,3,figsize=(18,6))
    ax[0].plot(t,E1,'b-', label="Quinn")
    ax[0].plot(t,E2,'m-', label="Boris")
    ax[0].set_xlim(0,40)
    ax[0].set_ylim(0.16-1e-7, 0.16+1e-7)
    ax[0].set_xlabel(r"$t[\Omega^{-1}]$")
    ax[0].set_ylabel(r"$E$")
    ax[1].plot(t,ds1,'b-')
    ax[1].plot(t,ds2,'m-')
    ax[1].set_xlim(0,40)
    ax[1].set_ylim(0,0.000014)
    ax[1].set_xlabel(r"$t[\Omega^{-1}]$")
    ax[1].set_ylabel(r"$\Delta s$")
    ax[2].loglog([1e-3,1e-2,1e-1], [err13,err12,err11], 'b--s')
    ax[2].loglog([1e-3,1e-2,1e-1], [err23,err22,err21], 'm--s')
    ax[2].set_xlim(5e-4,2e-1)
    ax[2].set_ylim(1e-6,1e-1)
    ax[2].set_xlabel(r"$\Delta t[\Omega^{-1}]$")
    ax[2].set_ylabel(r"$\left<\Delta s\right>$")
    fig.tight_layout()
    ax[0].legend()
    fig.savefig("starpar.pdf")
