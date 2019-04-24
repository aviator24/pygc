import matplotlib.pyplot as plt
import numpy as np

def SFR(sig):
    """ 
    Star formation rate surface density in the unit of M_sun kpc^-2 yr^-1
    presented in Ostriker & Shetty 2011.
    Gas surface density is in the unit of M_sun pc^-2
    """
    return 0.1 * (sig/100)**2

if __name__ == '__main__':
    # os11 law
    sig = np.logspace(-1,5)
    plt.loglog(sig, SFR(sig), 'k--', label="OS11")

    # Busch NGC 1808
    ds = np.loadtxt("/home/smoon/data/ringsf/1808.txt", dtype=str)
    plt.loglog(10**(ds[:,1].astype(float)), 10**(ds[:,2].astype(float)), 'ro', label="NGC1808")

    # Hsieh NGC 1097

    ds = np.loadtxt("/home/smoon/data/ringsf/1097.txt", dtype=str)
    sigerr = 170
    plt.loglog(ds[:,2].astype(float), ds[:,3].astype(float), 'bo', label="NGC1097")
#    XCO = 3e20
#    mh2 = 3.35e-24
#    Msun = 1.99e33
#    pc = 3.09e18
#    nu = 230.5e9
#    sigma = ds[:,1].astype(float)*(3e10/nu)**2/2/1.38e-16*XCO*mh2/Msun*pc**2
#    plt.loglog(sigma, ds[:,3].astype(float), '+', label="NGC1097")

    # Kennicutt 1998
    ds = np.loadtxt("/home/smoon/data/ringsf/ks_disk.txt", dtype=str)
    plt.loglog(10**(ds[:,1].astype(float)), 10**(ds[:,2].astype(float)), 'ks', label="K98 (normal)")
    ds = np.loadtxt("/home/smoon/data/ringsf/ks_sb.txt", dtype=str)
    plt.loglog(10**(ds[:,1].astype(float)), 10**(ds[:,2].astype(float)), 'k+', label="K98 (starbursts)")

    plt.xlim(1e0,1e5)
    plt.xticks([1e0,1e1,1e2,1e3,1e4,1e5])
    plt.ylim(1e-4,1e3)
    plt.xlabel(r"$\log\Sigma_{\rm mol, gas}$")
    plt.ylabel(r"$\log\Sigma_{\rm SFR}$")
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("sfr.png")
