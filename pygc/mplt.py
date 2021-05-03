import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
import numpy as np
import pyathena as pa
import xarray as xr
u = pa.util.units.Units()

def set_xy_axis(s,axis):
    """ Set x and y axis, label, and plot limits

    Depending on the cut axis, set appropriate x and y directions,
    labels, and plot limits.
    """
    if axis=='z':
        x = 'x'
        y = 'y'
        xlabel = '$x$'
        ylabel = '$y$'
        xlim = (s.domain['le'][0], s.domain['re'][0])
        ylim = (s.domain['le'][1], s.domain['re'][1])
    elif axis=='y':
        x = 'x'
        y = 'z'
        xlabel = '$x$'
        ylabel = '$z$'
        xlim = (s.domain['le'][0], s.domain['re'][0])
        ylim = (s.domain['le'][2], s.domain['re'][2])
    elif axis=='x':
        x = 'y'
        y = 'z'
        xlabel = '$y$'
        ylabel = '$z$'
        xlim = (s.domain['le'][1], s.domain['re'][1])
        ylim = (s.domain['le'][2], s.domain['re'][2])
    return x, y, xlabel, ylabel, xlim, ylim

def proj(ax,s,ds,
         axis='z',title='',vmin=1e0,vmax=1e3,dat=xr.Dataset()):
    """Draw projection plot at given snapshot number

    Arguments:
        ax   : axes to draw
        s    : LoadSim object
        ds   : AthenaDataSet object
        axis : axis to project (default:'z')
        title: axes title (default:'')
        vmin : minimum imshow color level (default:1e0)
        vmax : maximum imshow color level (default:1e3)
    """
    # set plot attributes
    x, y, xlabel, ylabel, xlim, ylim = set_xy_axis(s,axis)

    # load data
    if not 'density' in dat:
        dat = dat.merge(ds.get_field(['density']))
    dx = dat[axis][1]-dat[axis][0]
    dat['surf'] = (dat.density*u.Msun*dx).sum(dim=axis)

    # draw
    (dat.surf).plot.imshow(ax=ax,norm=LogNorm(vmin,vmax),
        cmap='pink_r',
        cbar_kwargs={'label':'$\\Sigma\,[M_\\odot\,{\\rm pc}^{-2}]$'})
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')

def sliceplot(ax,s,ds,f='nH',axis='z',pos=0,title=''):
    """Draw slice plot at given snapshot number

    Arguments:
        ax   : axes to draw
        s    : LoadSim object
        ds   : AthenaDataSet object
        f    : field to draw
        axis : axis to slice (default:'z')
        pos  : the cut position (defulat = 0)
        title: axes title (default:'')
    """

    # set plot attributes
    x, y, xlabel, ylabel, xlim, ylim = set_xy_axis(s,axis)

    # load data
    dat = ds.get_slice(axis,f,pos=pos)

    # draw
    dat[f].plot.imshow(ax=ax,**ds.dfi[f]['imshow_args'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')

def quiver(ax,s,ds,which='B',
           axis='z',pos=0,avg=None,hw=None,nbin=8,
           title='',scale=1,color='red',dat=xr.Dataset()):
    # set plot attributes
    x, y, xlabel, ylabel, xlim, ylim = set_xy_axis(s,axis)
    if axis=='z':
        if which=='vel':
            vx = 'velocity1'
            vy = 'velocity2'
        elif which=='B':
            vx = 'cell_centered_B1'
            vy = 'cell_centered_B2'
        else:
            raise KeyError('which = (vel|B)')
    elif axis=='y':
        if which=='vel':
            vx = 'velocity1'
            vy = 'velocity3'
        elif which=='B':
            vx = 'cell_centered_B1'
            vy = 'cell_centered_B3'
        else:
            raise KeyError('which = (vel|B)')
    elif axis=='x':
        if which=='vel':
            vx = 'velocity2'
            vy = 'velocity3'
        elif which=='B':
            vx = 'cell_centered_B2'
            vy = 'cell_centered_B3'
        else:
            raise KeyError('which = (vel|B)')

    # load data
    if not vx in dat:
        dat = dat.merge(ds.get_field([vx,vy]))
    if avg is None:
        # slice
        dat = dat.sel(method='nearest', **{axis:pos})
    elif avg=='volume':
        # volume average along the given axis within [-hw,hw]
        if hw is None:
            raise KeyError("set hw = (half width for volume average)")
        else:
            dat = dat.sel(**{axis:slice(-hw,hw)}).mean(dim=axis)
    elif avg=='mass':
        # mass-weighted average along the given axis
        if not 'density' in dat:
            dat = dat.merge(ds.get_field('density'))
        dat = dat.weighted(dat.density).mean(dim=axis)
    # degrade resolution
    x_ = dat[x].coarsen({x:nbin}).mean()
    y_ = dat[y].coarsen({y:nbin}).mean()
    vx_ = dat[vx].coarsen({x:nbin,y:nbin}).mean()
    vy_ = dat[vy].coarsen({x:nbin,y:nbin}).mean()

    # draw
    ax.quiver(x_.data, y_.data, vx_, vy_,
        scale=scale, scale_units='x', color=color)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
   
def clusters(ax, cl, m0=2e2, agemax=40):
    """Draw scatterplot of star clusters

    Arguments:
        ax   : axes to draw
        cl   : DataFrame returned from load_starpar_vtk
        m0   : normalization of the symbol size
        agemax : max age; should be consistent with cl
    """

    stars = ax.scatter(cl.x1, cl.x2, marker='o', s=np.sqrt(cl.mass/m0), c=cl.mage,
            edgecolor='black', linewidth=0.3, vmax=agemax, vmin=0, cmap='cool_r', zorder=2, alpha=0.75)    
    return stars

def hst_Bmag(ax,s,**kwargs):
    hst = pa.read_hst(s.files['hst'])
    ax.semilogy(hst.time*u.Myr, np.sqrt(hst.B1**2+hst.B2**2+hst.B3**2)*u.muG)
    ax.set_ylim(1e-4,1e-1)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\left<|{\\bf B}|\\right>\,[\\mu G]$')
