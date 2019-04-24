import matplotlib.pyplot as plt
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
    return np.sqrt(mass/1000.)

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
    legend=ax.legend(s,label,scatterpoints=1,loc=2,ncol=3,fontsize='small',bbox_to_anchor=(0.0, 1.1), frameon=False)
    
    return legend

def draw(js, je, joined=False):
    unit=pa.set_units(muH=1.4271)
    codemass=unit['mass']
    codetime=unit['time']
    codelength=unit['length']
    codevel=unit['velocity']
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
        
        d=ds.read_all_data('density')
        surfxy=d.sum(axis=0)*dx[2]*codemass.value/codelength.value**2
        surfxz=d.sum(axis=1)*dx[1]*codemass.value/codelength.value**2
        
        # This line reads in star particle data
        sp_file=ds.starfile
        sp=pa.read_starvtk(sp_file)
        
        # let's combine surface density map with star particles.
        
        fig,ax=plt.subplots(1,2,figsize=(24,8))
        
        plt.rcParams['font.size']=12
        
        ax[0].set_xlim(xmin[0],xmax[0])
        ax[0].set_ylim(xmin[1],xmax[1])
        ax[1].set_xlim(xmin[0],xmax[0])
        ax[1].set_ylim(xmin[2],xmax[2])
      
        # xy projection
        im=ax[0].imshow(surfxy,norm=LogNorm(),origin='lower',zorder=0,extent=xyextent,cmap='pink_r',clim=[1.e-1,1.e3])
        cbar=plt.colorbar(im,ax=ax[0])
        cbar.set_label(r'$\Sigma [M_{\odot} {\rm pc}^{-2}]$')
        ax[0].set_xlabel(r'$x [{\rm pc}]$')
        ax[0].set_ylabel(r'$y [{\rm pc}]$')
        
        #This uses the star particle plotting and legend
        cl=sp_plot(ax[0],sp,proj='z')
        leg=sp_legend(ax[0])
        
        # Now adding the colorbar for cluster particles
        cax1 = fig.add_axes([0.55, 0.93, 0.2, 0.02]) # [left, bottom, width, height]
        cbar=plt.colorbar(cl,ticks=[0,20,40],cax=cax1,orientation='horizontal')
        cbar.ax.set_title(r'$age [Myr]$',size='small')
      
        # xz projection
        im=ax[1].imshow(surfxz,norm=LogNorm(),origin='lower',zorder=0,extent=xzextent,cmap='pink_r',clim=[1.e0,1.e4])
        cbar=plt.colorbar(im,ax=ax[1])
        cbar.set_label(r'$\Sigma [M_{\odot} {\rm pc}^{-2}]$')
        ax[1].set_xlabel(r'$x [{\rm pc}]$')
        ax[1].set_ylabel(r'$z [{\rm pc}]$')
        
        #This uses the star particle plotting and legend
        cl=sp_plot(ax[1],sp,proj='y')
       
        ax[1].text(-400,300,r'$t=$'+'{:.1f} Myr'.format(0.1*i))
        # You can save the figure by uncommenting the following command
        fig.savefig('surfmap_{:04d}.png'.format(i),bbox_inches='tight')
        plt.close(fig)

def tmp():
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

if __name__ == '__main__':
    print("running the script: {0}".format(sys.argv[0]))
    print("Number of arguments: {0}".format(len(sys.argv)))
    print("draw images from time {0} to {1}".format(sys.argv[1],sys.argv[2]))
    draw(int(sys.argv[1]),int(sys.argv[2]),joined=False)
