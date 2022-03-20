%load_ext autoreload

import sys, os
sys.path.insert(0, '../')
sys.path.insert(0, '../python_src/')
sys.path.insert(0, '../../lib_persistent_homology/')
sys.path.insert(0, '../../lib_persistent_homology/python_src/')
sys.path.insert(0, '../../lib_network_tuning/')
sys.path.insert(0, '../../lib_network_tuning/python_src/')
sys.path.insert(0, '../../lib_softness/python_src/')


import numpy as np
import scipy as sp
import pandas as pd
import numpy.ma as ma
import numpy.linalg as la
import numpy.random as rand
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import pickle
import glob
import netCDF4
import matplotlib.patches as mpatches
import matplotlib.collections as mc
import sklearn as skl
from sklearn import svm
import time
from sklearn.model_selection import cross_val_score
import gsd.pygsd
import gsd.hoomd
import gsd.fl

import network_solver as ns
import network_plot as nplot
import softness as soft
import phom

import alpha_complex_algs as algs

%autoreload 2



DIM = 2
seed = 5
index = 0

# src_dir = "/data/p_ajliu/ridout/nasshear/{0:d}d/jason/".format(DIM)
src_dir = "/data/p_ajliu/ridout/nasshear/{0:d}d/jason/N00512/".format(DIM)
# src_dir = "/data/p_ajliu/ridout/nasshear/{0:d}d/jason/N16384/".format(DIM)
# src_dir = "/data/p_ajliu/ridout/nasshear/{0:d}d/jason/Lp-2.0/".format(DIM)


with gsd.fl.GSDFile(name = src_dir +  "{0:d}_dropinfo.gsd".format(seed), mode='rb') as decoFlow:
# with gsd.fl.GSDFile(name = src_dir +  "dropinfo_{0:d}.gsd".format(seed), mode='rb') as decoFlow:

    M = decoFlow.nframes
    
    time = decoFlow.read_chunk(frame=index,name='tc')[0]
    
    print(M, time)

    mode = np.array(decoFlow.read_chunk(frame=index,name='rawMode'), float)



state = netCDF4.Dataset(src_dir + "{0:d}_state.nc".format(seed), 'r')

(embed, rad2) = soft.tri.get_configuration(state, time)

print(embed.box_mat)

comp = soft.tri.construct_triangulation(embed, rad2)

print(comp.ncells)

D2min = soft.tri.calc_D2min(comp, mode, embed)

D2mindiff = soft.tri.compute_edge_differences(D2min,embed,rad2)

#now structure functions
allp = np.arange(NP).astype(int)
df_vert = soft.strfunc.get_go_counts(allp, comp, embed, rad2, 8)


