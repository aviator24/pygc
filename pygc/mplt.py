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

def proj(ax,path,num,
         axis='z',title='',vmin=1e0,vmax=1e3):
    """Draw projection plot at given snapshot number

    Arguments:
        ax   : axes to draw
        path : path to the base directory of the simulation
        num  : snapshot number
        axis : axis to project (default:'z')
        title: axes title (default:'')
        vmin : minimum imshow color level (default:1e0)
        vmax : maximum imshow color level (default:1e3)
    """
    # load simulation
    s = pa.LoadSim(path, verbose=False)
    ds = s.load_vtk(num)

    # set plot attributes
    x, y, xlabel, ylabel, xlim, ylim = set_xy_axis(s,axis)

    # load data
    dat = ds.get_field(['density'])
    dx = dat[axis][1]-dat[axis][0]
    dat['surf'] = (dat.density*dx).sum(dim=axis)

    # draw
    (dat.surf*u.Msun).plot.imshow(ax=ax,norm=LogNorm(vmin,vmax),
        cmap='pink_r',
        cbar_kwargs={'label':'$\\Sigma\,[M_\\odot\,{\\rm pc}^{-2}]$'})
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')

def slice(ax,path,num,f='nH',axis='z',pos=0,title=''):
    # load simulation
    s = pa.LoadSim(path, verbose=False)
    ds = s.load_vtk(num)

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

def quiver(ax,path,num,which='vel',
           axis='z',pos=0,method='nearest',avg=None,hw=None,nbin=8,
           title='',scale=1e-7,color='red'):
    # load simulation
    s = pa.LoadSim(path, verbose=False)
    ds = s.load_vtk(num)

    # set plot attributes
    x, y, xlabel, ylabel, xlim, ylim = set_xy_axis(s,axis)
    if axis=='z':
        if which=='vel':
            vx = 'vx'
            vy = 'vy'
        elif which=='B':
            vx = 'Bx'
            vy = 'By'
        else:
            raise KeyError('which = (vel|B)')
    elif axis=='y':
        if which=='vel':
            vx = 'vx'
            vy = 'vz'
        elif which=='B':
            vx = 'Bx'
            vy = 'Bz'
        else:
            raise KeyError('which = (vel|B)')
    elif axis=='x':
        if which=='vel':
            vx = 'vy'
            vy = 'vz'
        elif which=='B':
            vx = 'By'
            vy = 'Bz'
        else:
            raise KeyError('which = (vel|B)')

    # load data
    dat = ds.get_field([vx,vy])
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
    
def hst_Bmag(ax,paths,**kwargs):
    for path in paths:
        # load simulation
        s = pa.LoadSim(path, verbose=False)
        hst = pa.read_hst(s.files['hst'])
        ax.semilogy(hst.time*u.Myr, np.sqrt(hst.B1**2+hst.B2**2+hst.B3**2)*u.muG)
    ax.set_ylim(1e-4,1e-1)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\left<|{\\bf B}|\\right>\,[\\mu G]$')
