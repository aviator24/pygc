import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    ds = np.loadtxt("/home/smoon/data/ringsf/benedict02.txt")
    plt.hist(ds[:,-1]*1e4, histtype='step')
    plt.xlabel(r"$M_{\rm NRC}$")
    plt.ylabel(r"$N$")
    plt.xlim(1e3,1e5)
    plt.ylim(1e0,1e2)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("tmp.png")
