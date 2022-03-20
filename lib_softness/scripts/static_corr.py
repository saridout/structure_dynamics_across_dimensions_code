import pickle
import json
import argparse
import sys

import numpy as np
import scipy.stats as stats
import pandas as pd
import netCDF4
import gsd.pygsd
import gsd.hoomd
import gsd.fl

import softness as soft
import phom

from netCDF4 import Dataset
sys.path.append("/home/ridout/jscripts")
import corr
import iowrappers as io

parser = argparse.ArgumentParser(description="Get contributions of a single frame to softness and mode correlation functions.")
parser.add_argument("--basedir", help="Directory to look in for states.")
parser.add_argument("--trainfile", help="File to look in for hyperplane")
parser.add_argument("--seed", help="Database number.")
parser.add_argument("--mode", help="Run mode. Use `test' to do a quick test.")
parser.add_argument("--frame", help="Which state to consider")
parser.add_argument("--df",  help="Filename to load df of descriptors instead of loading the state and triangulating.")
ARGS = parser.parse_args()

def load_frame(args={}):
    filename = args.basedir + "/state_d"+args.seed+".nc"
    state = Dataset(filename)
    return soft.tri.get_configuration(state, int(args.frame)) 

def corrfunc_frame(args={}):
    embed, rad2 = load_frame(args)
    with open(args.trainfile, "rb") as f:
        temp = pickle.load(f)
        try:
            clfs = temp["SV_CLF"]
        except:
            clfs = temp["clf"] #new files...yeah, bad inconsistency.
            x_col = temp["x_col"]
    try:
       with open(args.df, "rb") as f:
           df_gaps = pickle.load(f)
       df_gaps = df_gaps[int(df_gaps["time"])==int(args.frame)]
    except:
        comp = soft.tri.construct_triangulation(embed, rad2)
        (rattlers, comp, embed, rad2) = soft.tri.remove_rattlers(comp, embed, rad2)
        allp = np.array(range(comp.ndcells[0]))
        df_gaps = soft.strfunc.get_go_counts(allp, comp, embed, rad2, 8)
   
        x_col = df_gaps.columns.values[2:]
        x_col = np.sort(x_col)
        x_col = np.insert(x_col, 0, 'particle_type')
        print(x_col)
        df_gaps.sort_values('particle', inplace=True)
        df_gaps.reset_index(drop=True, inplace=True)

    X1 = df_gaps[x_col[:3]].values
    X8 = df_gaps[x_col[:24]].values
    S1 = clfs[0].decision_function(X1)
    S8 = clfs[7].decision_function(X8)
    DIM = len(embed.box_mat)

    fields = np.zeros((2, len(S1)))
    fields[0,:] = S1
    fields[1,:] = S8
    fields = fields.transpose()
    #now compute
    pos = np.zeros((len(S1),DIM))
    for i in range(len(S1)):
        pos[i] = embed.get_vpos(i)
    classes = (rad2 == np.max(rad2))
    pair_counts, prod_sum, avg_sum = corr.compute_brute_correlations_classes(pos=pos, box=embed.box_mat, r_cut = 0.5*embed.box_mat[0,0], virtual=True, fields = fields, bin_spacing = 0.1, classes = classes) 



    #now build a df
    key = [(args.seed, args.frame)]
    output = {"frame":key }
    output["pair_counts"] = [pair_counts]
    output["S1_prod_sum"] = [prod_sum[:,0]]
    output["S1_avg_sum"] = [avg_sum[:,0]]
    output["S8_prod_sum"] = [prod_sum[:,1]]
    output["S8_avg_sum"] = [avg_sum[:,1]]

    df = pd.DataFrame(output)
    df.set_index("frame", inplace=True)
    outfilename = args.basedir+"/corrfuncs.pkl"
    io.pd_concat_on_disk(df , outfilename)
    
corrfunc_frame(args=ARGS) 
