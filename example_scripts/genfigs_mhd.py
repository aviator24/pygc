#!/usr/bin/env python
"""Create 2x2 panel plot showing the gas surface density and star particles

  Typical usage example:

  genfigs.py `pwd` 200 210

  Written by Sanghyuk Moon
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
from pygc import mplt
import pyathena as pa
from pyathena.util.units import Units
u = Units()

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to the simulation data')
parser.add_argument('ns', type=int, help='start number')
parser.add_argument('ne', type=int, help='end number')
parser.add_argument('--step', default=1, type=int, help='increment')
args = parser.parse_args()

s = pa.LoadSim(args.path)
for num in np.arange(args.ns, args.ne+1, args.step):
    ds = s.load_vtk(num)
    dat = ds.get_field(['density','cell_centered_B'])
    dat['Bmag'] = np.sqrt(dat.cell_centered_B1**2
                          + dat.cell_centered_B2**2
                          + dat.cell_centered_B3**2)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(18,14))
    cax1 = mplt.get_cax(ax1, 'right', '5%', pad='1%')
    cax2 = mplt.get_cax(ax2, 'right', '5%', pad='1%')
    cax3 = mplt.get_cax(ax3, 'right', '5%', pad='1%')
    cax4 = mplt.get_cax(ax4, 'right', '5%', pad='1%')
    
    # surface density in x-y plane
    surf = mplt.proj(ax1, s, ds, dat=dat, vmin=1e-2, add_colorbar=False)
    plt.colorbar(surf, cax=cax1, label=r'$\Sigma\,[M_\odot\,{\rm pc}^{-2}]$')
    
    # magnetic field in x-y plane
    Bx_w = dat.cell_centered_B1.weighted(dat.density).mean(dim='z')*u.muG
    By_w = dat.cell_centered_B2.weighted(dat.density).mean(dim='z')*u.muG
    Bmag_w = dat.Bmag.weighted(dat.density).mean(dim='z')*u.muG
    mask = (Bmag_w>1e0)
    mag = Bmag_w.plot.imshow(ax=ax2, norm=LogNorm(1e0,1e3), cmap='plasma', add_colorbar=False)
    plt.colorbar(mag, cax=cax2, label=r'$|\bf B|\,[\mu G]$')
    ax2.streamplot(dat.x.data, dat.y.data, Bx_w.where(mask).data, By_w.where(mask).data, density=2, color='c')
    
    # surface density in x-z plane 
    surf = mplt.proj(ax3, s, ds, dat=dat, axis='y', vmin=1e-2, vmax=1e4, add_colorbar=False)
    plt.colorbar(surf, cax=cax3, label=r'$\Sigma\,[M_\odot\,{\rm pc}^{-2}]$')
    
    # magnetic field in x-z plane
    Bx_w = dat.cell_centered_B1.weighted(dat.density).mean(dim='y')*u.muG
    Bz_w = dat.cell_centered_B2.weighted(dat.density).mean(dim='y')*u.muG
    Bmag_w = dat.Bmag.weighted(dat.density).mean(dim='y')*u.muG
    mask = (Bmag_w>1e0)
    mag = Bmag_w.plot.imshow(ax=ax4, norm=LogNorm(1e0,1e3), cmap='plasma', add_colorbar=False)
    plt.colorbar(mag, cax=cax4, label=r'$|\bf B|\,[\mu G]$')
    ax4.streamplot(dat.x.data, dat.z.data, Bx_w.where(mask).data, Bz_w.where(mask).data, density=2, color='c')
    
    for ax in (ax1,ax2,ax3,ax4):
        ax.set_aspect('equal')
        ax.set_xlim(-1024,1024)
        ax.set_ylim(-1024,1024)
        ax.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax2.set_ylabel('$y$')
    ax3.set_ylabel('$z$')
    ax4.set_ylabel('$z$')
    
    fig.tight_layout()

    # add legend
    ax1.set_title(r'$t = {:.1f}$'.format(ds.domain['time']*u.Myr) +
                                         r'$\,{\rm Myr}$', fontsize=25)
    # save figure
    opath = Path('/home/sm69/figures-all/{}'.format(s.basename))
    opath.mkdir(parents=False, exist_ok=True)
    fname = 'projections.{:04d}.png'.format(num)
    fig.savefig(opath.joinpath(fname), bbox_inches='tight')
    plt.close(fig)
