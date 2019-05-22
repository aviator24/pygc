import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import constants as c
from astropy import units as u
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from pyathena import cooling,AthenaDataSet,set_units

def func(T,gm,dx):
    return T**1.5*np.exp(-92/T)-0.6*gm*dx**2

if __name__ == '__main__':
    cf=cooling.coolftn()
    coolft=interp1d(cf.temp,cf.cool)
    heatft=interp1d(cf.temp,cf.heat)
    muft=interp1d(cf.temp,cf.temp/cf.T1)

    heat_ratio=[1e0,1e1,1e2,1e3]
    muH=1.4271

    N=1000
    n=np.logspace(-1,5,N)
    Teq=np.zeros(N)
    fig,ax=plt.subplots(1,2,figsize=(16,6))

    ax[0].loglog(n,n*184,'k:')
    ax[1].loglog(n,184*np.ones(len(n)),'k:')
    for j in range(4):
        for i in range(N):
            try:
                Teq[i]=bisect(lambda x: heat_ratio[j]*heatft(x)/n[i]-coolft(x)
                        ,12.95,1e4)
            except:
                Teq[i]=np.nan
        Peq=n*Teq
        ax[0].loglog(n,Peq,'k-')
        ax[1].loglog(n,Teq,'k-')
    dxs=[8*u.pc,4*u.pc,2*u.pc]
    Teq=np.logspace(np.log10(12.95),5)
    cs2=c.k_B*(Teq*u.K)/muft(Teq)/c.m_p
    for dx in dxs:
        nth=(8.86/np.pi*cs2/c.G/dx**2/muH/c.m_p).to(u.cm**-3).value
        ax[1].loglog(nth,Teq,'k--')

    dx = 600*u.pc/128
    nth=100*(8.86/np.pi*cs2/c.G/dx**2/muH/c.m_p).to(u.cm**-3).value
    ax[1].loglog(nth,Teq,'b--')

    dx = 600*u.pc/256
    nth=10*(8.86/np.pi*cs2/c.G/dx**2/muH/c.m_p).to(u.cm**-3).value
    ax[1].loglog(nth,Teq,'r--')

    ax[0].set_xlim(1e-1,1e5)
    ax[0].set_ylim(1e3,1e7)
    ax[0].set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax[0].set_ylabel(r'$P/k_{\rm B}\,[{\rm K\,cm^{-3}}]$')
    ax[1].set_xlim(1e-1,1e5)
    ax[1].set_ylim(1e1,1e5)
    ax[1].set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax[1].set_ylabel(r'$T\,[{\rm K}]$')
    ax[0].text(5e1,2e4,
            r"$\Gamma=\Gamma_0=2\times 10^{-26}\,{\rm erg\,s^{-1}}$",
            fontsize=13,rotation=37)
    ax[0].text(5e2,8e4,r"$\Gamma=10\Gamma_0$",fontsize=13,rotation=40)
    ax[0].text(3e3,7e5,r"$\Gamma=10^2\Gamma_0$",fontsize=13,rotation=38)
    ax[0].text(0.8e2,3.3e6,r"$\Gamma=10^3\Gamma_0$",fontsize=13,rotation=48)
    ax[0].text(0.8e1,0.8e4,r"$T=184\,{\rm K}$",fontsize=13,rotation=52)
    ax[1].text(3e3,8e3,r"$\Delta x=8\,{\rm pc}$",fontsize=13,rotation=51)
    ax[1].text(6e3,4e3,r"$\Delta x=4\,{\rm pc}$",fontsize=13,rotation=51)
    ax[1].text(1.2e4,2e3,r"$\Delta x=2\,{\rm pc}$",fontsize=13,rotation=51)
    labels=[r"$\Gamma/\Gamma_0=1$",r"$\Gamma/\Gamma_0=10$",
            r"$\Gamma/\Gamma_0=10^2$",r"$\Gamma/\Gamma_0=10^3$"]
    fig.savefig("phase.pdf",bbox_inches='tight')

#    Ta=10
#    Tb=1e5
#    gm=1
#    muH=1.4271
#    dx=np.logspace(-1,2,100)
#    Tth=np.zeros(100)
#    dth=np.zeros(100)
#    for i in range(100):
#        Tth[i] = bisect(func,Ta,Tb,args=(gm,dx[i]))
#        dth[i] = ((8.86/np.pi*c.k_B*(Tth[i])*u.K/muH/c.m_p/c.G/((dx[i])*u.pc)**2)/(muH*c.m_p)).cgs.value
#    plt.loglog(dx,Tth,'b-')
#    plt.loglog(dx,dth,'g-')
#
#    Ta=10
#    Tb=1e5
#    gm=1000
#    muH=1.4271
#    dx=np.logspace(-1,2,100)
#    Tth=np.zeros(100)
#    dth=np.zeros(100)
#    for i in range(100):
#        Tth[i] = bisect(func,Ta,Tb,args=(gm,dx[i]))
#        dth[i] = ((8.86/np.pi*c.k_B*(Tth[i])*u.K/muH/c.m_p/c.G/((dx[i])*u.pc)**2)/(muH*c.m_p)).cgs.value
#    plt.loglog(dx,Tth,'b--')
#    plt.loglog(dx,dth,'g--')
#
#    plt.xlim(1e-1,1e2)
#    plt.ylim(1e0,1e7)
#    plt.tight_layout()
#    plt.savefig("th.pdf")
