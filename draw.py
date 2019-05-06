import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import astropy.constants as c
import astropy.units as u
from matplotlib.colors import LogNorm
import pyathena as pa
import sys
from pyathena import set_units

def mass_norm(mass):
    '''
    Mass normlization function to determine symbol size
    This should be called both in sp_plot and sp_legend for the consistent result
    '''
    return np.sqrt(mass/100.)

def sp_plot(ax,sp,proj=None):
    '''
    This function plots star particles 
    '''
    unit=pa.set_units(muH=1.4271)
    tunit_Myr=unit['time'].to('Myr').value
    munit_Msun=unit['mass'].to('Msun').value
     
    young_sp=sp[sp['age']*tunit_Myr < 40.]
    runaway=young_sp[young_sp['mass'] == 0]
    young_cluster=young_sp[young_sp['mass'] != 0]
    
    mass=young_cluster['mass']*munit_Msun
    age=young_cluster['age']*tunit_Myr
    
    if (proj=='z'): 
        cl=ax.scatter(young_cluster['x1'],young_cluster['x2'],marker='o',s=mass_norm(mass),c=age,edgecolor='black', linewidth=1, vmax=40,vmin=0,cmap=plt.cm.cool_r,zorder=2)
#        ax.scatter(runaway['x1'],runaway['x2'],marker='.',color='k',zorder=1)
    elif (proj=='y'):
        cl=ax.scatter(young_cluster['x1'],young_cluster['x3'],marker='o',s=mass_norm(mass),c=age,edgecolor='black', linewidth=1, vmax=40,vmin=0,cmap=plt.cm.cool_r,zorder=2)
#        ax.scatter(runaway['x1'],runaway['x3'],marker='.',color='k',zorder=1)
    else:
        raise Exception("The projection direction should be given") 
    
    return cl

def sp_legend(ax,ref_mass=[1.e4,1.e5,1.e6]):
    ext=ax.images[0].get_extent()

    #plot particle references outside of the domain of interest
    s=[]
    label=[]
    for mass in ref_mass:
        s.append(ax.scatter(ext[1]*2,ext[3]*2,s=mass_norm(mass),color='k',alpha=.5))
        label.append(r'$10^%d M_\odot$' % np.log10(mass))
    ax.set_xlim(ext[0],ext[1])
    ax.set_ylim(ext[2],ext[3])
    legend=ax.legend(s,label,scatterpoints=1,loc=2,ncol=3,bbox_to_anchor=(0.0, 1.1), frameon=False)
    
    return legend

def draw_all(js, je, joined=False):
    unit=pa.set_units(muH=1.4271)
    for i in range(js,je+1):
        if joined:
            ds=pa.AthenaDataSet('../gc.{:04d}.vtk'.format(i))
        else:
            ds=pa.AthenaDataSet('../id0/gc.{:04d}.vtk'.format(i))
        #This is domain information
        xmin=ds.domain['left_edge']
        xmax=ds.domain['right_edge']
        dx=ds.domain['dx']
        Nx=ds.domain['Nx']
        
        # set up cell centered coordinates
        x=np.arange(xmin[0],xmax[0],dx[0])+0.5*dx[0]
        y=np.arange(xmin[1],xmax[1],dx[1])+0.5*dx[1]
        z=np.arange(xmin[2],xmax[2],dx[2])+0.5*dx[2]
        
        #This sets up for image plots based on the domain physical size
        xyextent=[xmin[0],xmax[0],xmin[1],xmax[1]]
        xzextent=[xmin[0],xmax[0],xmin[2],xmax[2]]
        yzextent=[xmin[1],xmax[1],xmin[2],xmax[2]]

        dx=dx*unit['length']
        x=x*unit['length']
        y=y*unit['length']
        z=z*unit['length']
        
        rho=(ds.read_all_data('density')*unit['density']).to(u.M_sun/u.pc**3)
        den=(rho/unit['muH']).to(u.cm**-3)
        surfxy=((rho*dx[2]).sum(axis=0)).to(u.M_sun/u.pc**2)
        surfxz=((rho*dx[1]).sum(axis=1)).to(u.M_sun/u.pc**2)
        T=ds.read_all_data('temperature')*unit['temperature']
        prs=ds.read_all_data('pressure')*unit['pressure']
        pok=(prs/c.k_B).to(u.K/u.cm**3)
        
        # This line reads in star particle data
        sp_file=ds.starfile
        sp=pa.read_starvtk(sp_file)
        
        fig = plt.figure(figsize=(24,15))
        gs = gridspec.GridSpec(2,2,figure=fig,height_ratios=[2,1],hspace=0.1)
        ax=np.ndarray((2,2),dtype=object)
        ax[0,0] = fig.add_subplot(gs[0,0])
        ax[0,1] = fig.add_subplot(gs[0,1])
        ax[1,0] = fig.add_subplot(gs[1,0])
        ax[1,1] = fig.add_subplot(gs[1,1])
        
        ax[0,0].set_xlim(xmin[0],xmax[0])
        ax[0,0].set_ylim(xmin[1],xmax[1])
        ax[0,1].set_xlim(xmin[0],xmax[0])
        ax[0,1].set_ylim(xmin[1],xmax[1])
        ax[1,0].set_xlim(xmin[0],xmax[0])
        ax[1,0].set_ylim(xmin[2],xmax[2])
        ax[1,1].set_xlim(xmin[0],xmax[0])
        ax[1,1].set_ylim(xmin[2],xmax[2])

        # xy projection
        im=ax[0,0].imshow(surfxy,norm=LogNorm(),origin='lower',zorder=0,
                extent=xyextent,cmap='pink_r',clim=[1.e-1,1.e3])
        cbar=plt.colorbar(im,ax=ax[0,0])
        cbar.set_label(r'$\Sigma\,[M_{\odot} {\rm pc}^{-2}]$')
        ax[0,0].set_ylabel(r'$y\,[{\rm pc}]$')
        cl=sp_plot(ax[0,0],sp,proj='z')
        leg=sp_legend(ax[0,0])
      
        # xy slice
        im=ax[0,1].imshow(den[Nx[2]>>1,:,:],norm=LogNorm(),origin='lower',zorder=0,
                extent=xyextent,cmap='BuPu',clim=[1.e-2,1.e4])
        cbar=plt.colorbar(im,ax=ax[0,1])
        cbar.set_label(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
        ax[0,1].set_ylabel(r'$y\,[{\rm pc}]$')

        # xz projection
        im=ax[1,0].imshow(surfxz,norm=LogNorm(),origin='lower',zorder=0,
                extent=xzextent,cmap='pink_r',clim=[1.e0,1.e4])
        cbar=plt.colorbar(im,ax=ax[1,0])
        cbar.set_label(r'$\Sigma\,[M_{\odot} {\rm pc}^{-2}]$')
        ax[1,0].set_xlabel(r'$x\,[{\rm pc}]$')
        ax[1,0].set_ylabel(r'$z\,[{\rm pc}]$')
        cl=sp_plot(ax[1,0],sp,proj='y')

        # xz slice
        im=ax[1,1].imshow(den[:,Nx[1]>>1,:],norm=LogNorm(),origin='lower',zorder=0,
                extent=xzextent,cmap='BuPu',clim=[1.e-2,1.e4])
        cbar=plt.colorbar(im,ax=ax[1,1])
        cbar.set_label(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
        ax[1,1].set_xlabel(r'$x\,[{\rm pc}]$')
        ax[1,1].set_ylabel(r'$z\,[{\rm pc}]$')
    

        # annotations
        cax1 = fig.add_axes([0.15, 0.93, 0.25, 0.015]) # [left, bottom, width, height]
        cbar=plt.colorbar(cl,ticks=[0,20,40],cax=cax1,orientation='horizontal')
        cbar.ax.set_title(r'$age\,[\rm Myr]$')
        ax[0,1].text(-60,600,r'$t={:.1f}\,\rm Myr$'.format(0.1*i))
        fig.tight_layout()
        fig.savefig('all_{:04d}.png'.format(i),bbox_inches='tight')
        plt.close(fig)

        # phase diagram (n-P,T-P)
        fig, ax = plt.subplots(1,2,figsize=(22,10))
        ax[0].hexbin(den.flatten(),pok.flatten(),xscale='log',yscale='log'
                ,cmap='Greys',mincnt=1,bins='log',gridsize=(100,100),
                C=den.flatten(),reduce_C_function=np.sum)
        ax[1].hexbin(den.flatten(),T.flatten(),xscale='log',yscale='log'
                ,cmap='Greys',mincnt=1,bins='log',gridsize=(100,100),
                C=den.flatten(),reduce_C_function=np.sum)
        ax[0].set_xlim(1e-3,1e4)
        ax[0].set_ylim(1e1,1e7)
        ax[1].set_xlim(1e-3,1e4)
        ax[1].set_ylim(1e1,1e7)

        ax[0].set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
        ax[0].set_ylabel(r'$P/k_{\rm B}\,[{\rm K\,cm^{-3}}]$')
        ax[1].set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
        ax[1].set_ylabel(r'$T\,[{\rm K}]$')
        ax[0].text(2e-5,3e6,r'$t={:.1f}\,\rm Myr$'.format(0.1*i))
        fig.tight_layout()
        fig.savefig('phase_{:04d}.png'.format(i),bbox_inches='tight')
        plt.close(fig)

def tmporary():
    i=1
    ds=pa.AthenaDataSet('../id0/gc.{:04d}.vtk'.format(i))
    #This is domain information
    xmin=ds.domain['left_edge']
    xmax=ds.domain['right_edge']
    dx=ds.domain['dx']
    Nx=ds.domain['Nx']
    
    # set up cell centered coordinates
    x=np.arange(xmin[0],xmax[0],dx[0])+0.5*dx[0]
    y=np.arange(xmin[1],xmax[1],dx[1])+0.5*dx[1]
    z=np.arange(xmin[2],xmax[2],dx[2])+0.5*dx[2]
    
    #This sets up for image plots based on the domain physical size
    xyextent=[xmin[0],xmax[0],xmin[1],xmax[1]]
    xzextent=[xmin[0],xmax[0],xmin[2],xmax[2]]
    yzextent=[xmin[1],xmax[1],xmin[2],xmax[2]]
    
    d=ds.read_all_data('density')
    T=ds.read_all_data('temperature')

def draw_hst(Lx,Ly,Lz,tmax=50):
    """ Lx,Ly,Lz should be given in pc """
    vol = Lx*Ly*Lz
    print("volume = {:e} pc^3".format(vol))
    unit=pa.set_units(muH=1.4271)
    ds = np.loadtxt("../id0/gc.hst")
    t = ds[:,0]*unit['time']
    Mtot = ds[:,2]*vol*unit['mass']
    Pth = ds[:,38]*unit['pressure']
    Pturb = ds[:,40]*unit['pressure']
    Mw = ds[:,51]*vol*unit['mass']
    Mu = ds[:,52]*vol*unit['mass']
    Mc = ds[:,53]*vol*unit['mass']
    Ms = ds[:,57]*vol*unit['mass']
    sigsfr = ds[:,54]*u.M_sun/u.Myr/u.pc**2
    sfr = (sigsfr*Lx*Ly*u.pc**2).to(u.M_sun/u.yr)

    plt.plot(t, Mc+Mu, 'b-', label=r"$M_u+M_c$")
    plt.plot(t, Mw, 'r-', label=r"$M_w$")
    plt.plot(t, Mtot, 'k-', label=r"$M_{\rm tot}$")
    plt.plot(t, Ms, 'k--', label=r"$M_{\rm sp}$")
    plt.xlabel("time ["+r"${\rm Myr}$"+"]")
    plt.ylabel("mass ["+r"${M_\odot}$"+"]")
    plt.xlim(0,tmax)
    plt.ylim(0,5e7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mass.pdf")
    plt.clf()

    plt.plot(t, sfr, 'k-')
    plt.xlabel("time ["+r"${\rm Myr}$"+"]")
    plt.ylabel("star formation rate ["+r"$M_\odot\,{\rm yr}^{-1}$"+"]")
    plt.xlim(0,tmax)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig("sfr.pdf")
    plt.clf()

if __name__ == '__main__':
    print("running the script: {0}".format(sys.argv[0]))
    print("Number of arguments: {0}".format(len(sys.argv)))
    print("draw images from time {0} to {1}".format(sys.argv[1],sys.argv[2]))
#    density_projection(int(sys.argv[1]),int(sys.argv[2]),joined=False)
    draw_all(int(sys.argv[1]),int(sys.argv[2]),joined=True)
#    draw_hst(600,600,1200,tmax=350)
