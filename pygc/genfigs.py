#!/usr/bin/env python
"""Create 2x2 panel plot showing the gas surface density and star particles

  Typical usage example:

  genfigs.py `pwd` 200 210

  Written by Sanghyuk Moon
"""
import argparse
import matplotlib.pyplot as plt
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
    dat = ds.get_field(['density'])
    sp = s.load_starpar_vtk(num)
    if sp.size==0:
        starpar=False
    else:
        starpar=True
    if starpar:
        sp['mage'] *= u.Myr
        sp['mass'] *= u.Msun
        cl = sp[sp.mage<10]

    # create figure and axes 
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(18,14))
    cax1 = mplt.get_cax(ax1, 'right', '5%', pad='1%')
    cax2 = mplt.get_cax(ax2, 'right', '5%', pad='1%')
    cax3 = mplt.get_cax(ax3, 'right', '5%', pad='1%')
    cax4 = mplt.get_cax(ax4, 'right', '5%', pad='1%')

    # surface density in x-y plane
    surf = mplt.proj(ax1, s, ds, dat=dat, vmin=1e-2, add_colorbar=False)
    plt.colorbar(surf, cax=cax1, label=r'$\Sigma\,[M_\odot\,{\rm pc}^{-2}]$')

    # star particles in x-y plane
    mplt.proj(ax2, s, ds, dat=dat, vmin=1e-2, alpha=0.5, add_colorbar=False)
    if starpar:
        stars, ss, label = mplt.clusters(ax2, cl, m0=2e1, agemax=10,
                                         mass_labels=[1e3,1e4,1e5,1e6])
        plt.colorbar(stars, cax=cax2, label=r'${\rm age}\,[{\rm Myr}]$')

    # surface density in x-z plane 
    surf = mplt.proj(ax3, s, ds, dat=dat, axis='y', vmin=1e-2,
                     add_colorbar=False)
    plt.colorbar(surf, cax=cax3, label=r'$\Sigma\,[M_\odot\,{\rm pc}^{-2}]$')

    # star particles in x-z plane
    mplt.proj(ax4, s, ds, dat=dat, axis='y', vmin=1e-2, alpha=0.5,
              add_colorbar=False)
    if starpar:
        stars, ss, label = mplt.clusters(ax4, cl, axis='y', m0=2e1, agemax=10,
                                         mass_labels=[1e3,1e4,1e5,1e6])
        plt.colorbar(stars, cax=cax4, label=r'${\rm age}\,[{\rm Myr}]$')

    # add legend
    ax1.set_title(r'$t = {:.1f}$'.format(ds.domain['time']*u.Myr) +
                                         r'$\,{\rm Myr}$', fontsize=25)
    if starpar:
        ax2.legend(ss, label, scatterpoints=1, loc=2, ncol=4,
                   bbox_to_anchor=(-0.7, 1.15), frameon=False)
    for ax in (ax1,ax2,ax3,ax4):
        ax.set_aspect('equal')

    # save figure
    opath = Path('/home/sm69/figures-all/{}'.format(s.basename))
    opath.mkdir(parents=False, exist_ok=True)
    fname = 'projections.{:04d}.png'.format(num)
    fig.savefig(opath.joinpath(fname), bbox_inches='tight')
    plt.close(fig)
