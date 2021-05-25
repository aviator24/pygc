import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
import numpy as np
import pyathena as pa
import xarray as xr
u = pa.util.units.Units()

def proj(ax,s,ds,
         axis='z',title='',vmin=1e0,vmax=1e3,dat=xr.Dataset(),**kwargs):
    """Draw projection plot at given snapshot number

    Args:
        ax: axes to draw.
        s: LoadSim object.
        ds: AthenaDataSet object.
        axis: axis to project (default:'z').
        title: axes title (default:'').
        vmin: minimum imshow color level (default:1e0).
        vmax: maximum imshow color level (default:1e3).

    Returns:
        img: the image that goes into plt.colorbar

    Example:
        surf = proj(ax, s, ds, dat=dat, add_colorbar=False)
        plt.colorbar(surf, cax=cax, label=r'$\Sigma\,[M_\odot\,{\rm pc}^{-2}]$')
    """

    # set plot attributes
    x, y, xlabel, ylabel, xlim, ylim = _set_xy_axis(s,axis)

    # load data
    if not 'density' in dat:
        dat = dat.merge(ds.get_field(['density']))
    dx = dat[axis][1]-dat[axis][0]
    surf = (dat.density*u.Msun*dx).sum(dim=axis)

    # draw
    if 'add_colorbar' in kwargs:
        if (kwargs['add_colorbar'] == False):
            img = surf.plot.imshow(ax=ax,norm=LogNorm(vmin,vmax), cmap='pink_r',
                                   **kwargs)
        else:
            img = surf.plot.imshow(
                ax=ax,norm=LogNorm(vmin,vmax), cmap='pink_r',
                cbar_kwargs={'label':r'$\Sigma\,[M_\odot\,{\rm pc}^{-2}]$'},
                **kwargs)
    else:
        img = surf.plot.imshow(
            ax=ax,norm=LogNorm(vmin,vmax), cmap='pink_r',
            cbar_kwargs={'label':r'$\Sigma\,[M_\odot\,{\rm pc}^{-2}]$'},
            **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    return img

def sliceplot(ax,s,ds,f='nH',axis='z',pos=0,title=''):
    """Draw slice plot at given snapshot number

    Args:
        ax: axes to draw.
        s: LoadSim object.
        ds: AthenaDataSet object.
        f: field to draw.
        axis: axis to slice (default:'z').
        pos: the cut position (defulat = 0).
        title: axes title (default:'').
    """

    # set plot attributes
    x, y, xlabel, ylabel, xlim, ylim = _set_xy_axis(s,axis)

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
    x, y, xlabel, ylabel, xlim, ylim = _set_xy_axis(s,axis)
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
   
def clusters(ax, cl, m0=2e2, agemax=40, axis='z', mass_labels=[1e4,1e5,1e6]):
    """Draw scatterplot of star clusters colored by their age

    The size of the symbol scales with the star particle mass. The legends for
    the size reference can be created by, e.g.,
      ax.legend(ss, label, scatterpoints=1, loc=2, ncol=4,
                bbox_to_anchor=(-0.7, 1.15), frameon=False)
    Args:
        ax: axes to draw.
        cl: DataFrame returned from load_starpar_vtk.
        m0: normalization of the symbol size.
        agemax: max age; should be consistent with maximum age cut of cl.
        axis: axis to project.
        mass_labels: star particle mass labels.

    Returns:
        stars: scatterplot instance to be used for colorbar.
        ss, label: arguments that goes into plt.legend() for particle size.

    Example:
        stars, ss, label = clusters(ax, cl, mass_labels=[1e4,1e5,1e6,1e7])
    """

    if (axis=='z'):
        stars = ax.scatter(
            cl.x1, cl.x2, marker='o', s=np.sqrt(cl.mass/m0), c=cl.mage,
            edgecolor='black', linewidth=0.3, vmax=agemax, vmin=0,
            cmap='cool_r', zorder=2, alpha=0.75)
    elif (axis=='y'):
        stars = ax.scatter(
            cl.x1, cl.x3, marker='o', s=np.sqrt(cl.mass/m0), c=cl.mage,
            edgecolor='black', linewidth=0.3, vmax=agemax, vmin=0,
            cmap='cool_r', zorder=2, alpha=0.75)
    elif (axis=='x'):
        stars = ax.scatter(
            cl.x2, cl.x3, marker='o', s=np.sqrt(cl.mass/m0), c=cl.mage,
            edgecolor='black', linewidth=0.3, vmax=agemax, vmin=0,
            cmap='cool_r', zorder=2, alpha=0.75)

    ss = []
    label = []
    for mass in mass_labels:
        ss.append(ax.scatter(-2000, 2000, marker='o', s=np.sqrt(mass/m0),
                  c='k', linewidth=0.3, alpha=0.75))
        label.append(r'$10^{:d}\,M_\odot$'.format(int(np.log10(mass))))

    return stars, ss, label

def hst_Bmag(ax,s,**kwargs):
    hst = pa.read_hst(s.files['hst'])
    ax.semilogy(hst.time*u.Myr, np.sqrt(hst.B1**2+hst.B2**2+hst.B3**2)*u.muG)
    ax.set_ylim(1e-4,1e-1)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\left<|{\\bf B}|\\right>\,[\\mu G]$')

def get_cax(ax, position, size, pad=None):
    """Append a colorbar axes at the given axes

    Args:
        ax: the parent axes.
        position: ["left"|"right"|"bottom"|"top"]
        size: the size of the cax relative to the parent axes.

    Returns:
        cax: the colorbar axes.

    Example:
        get_cax(ax, 'right', '5%', pad='1%')
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes(position, size, pad=pad)
    return cax

def _set_xy_axis(s,axis):
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
