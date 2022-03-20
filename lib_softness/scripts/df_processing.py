#Load the 32 trajectory dfs containing relative D2min, etc.
#Determine softness by regression / local extrema.  Determine R^2, classification accuracy and percentile of global max.

import pickle
import argparse
import os
import re

import numpy as np
import pandas as pd

from scipy import stats

import softness.regressor as regressor
import softness.classifier as classifier
PARSER = argparse.ArgumentParser("Load descriptor dataframes, analyze regression softness.")
PARSER.add_argument(
    "-n", "--NP", help="Number of particles in system.", type=int)
PARSER.add_argument('--dim', help="Spatial dimension.", type=int, default=2)
PARSER.add_argument('--basedir', help = "/blah/blah/blah= , seed goes after = ")
PARSER.add_argument('--nogaps', action="store_true", help = "Do not use gaps in descriptors.")
PARSER.add_argument('--notype', action="store_true", help = "Do not use particle type in descriptors.")

ARGS = PARSER.parse_args()
print(ARGS.basedir)
def main(args):
    max_seed = 32
    df = pd.DataFrame()
    for seed in range(1,max_seed+1):
       filename = args.basedir+ str(seed) + "/descriptor_df.pkl"
       try:
           with open(filename, "rb") as f:
               df = pd.concat([df, pickle.load(f)] )
       except:
           print("No data: ", filename)

    #now dataframe is complete, prepare for analysis
    if args.notype:
        x_col = [(i, t) for i in range(1,9) for t in ["g","o"] ]
    elif args.nogaps:
        x_col = ["particle_type", ] + [(i, t) for i in range(1,9) for t in ["o"] ]
    else:
        x_col = ["particle_type", ] + [(i, t) for i in range(1,9) for t in ["g","o"] ]
    #do 8 regressions by distance
    reg = []   
    print(df.columns)
    print(x_col)
    print("regressions")
    for l in range(8):
        reg.append(regressor.get_regressor(df, x_col[:3+2*l], 'D2min_normed'))
    clf = []
    for l in range(8): 
        clf.append(classifier.get_classifier(df[df['local'] != 0], x_col[:3+2*l], 'local', clf_type="SV"))
      
    clf_dir = "/projects/ajliu/ridout/data/softsphere_hyplanes/"
    clf_output = {"x_col":x_col, "reg":reg, "clf":clf} 
    #god I hate doing this crap
    if re.search("memsshear", args.basedir) is not None:
        clffile = "memsshear_"
    else:
        clffile = "nasshear_"

    clffile += str(args.dim) + "d_N="+str(args.NP)+"_"

    if re.search("phi=.*_", args.basedir) is not None:
        clffile += re.search("phi=[0123456789\.]*_", args.basedir)[0][:-1]
    elif re.search("Lp=.*_", args.basedir) is not None:
        clffile += re.search("Lp=[0123456789\.]*_", args.basedir)[0][:-1]
    if args.notype:
        clffile += "_notype"
    elif args.nogaps:
        clffile += "_nogaps"
    print("clf:", clf_dir+clffile)
    with open(clf_dir+ clffile, "wb") as f:
       pickle.dump(clf_output, f)
    #post-training, we need to compute <percentile_GM> 
    #the correlation function would also probably be good to compute...but it potentially takes a very long time
    #so hold off on the correlation function for now!
    #we satisfy ourselves with the S distribution. Want to collect it in a way that handles "almost discrete" case well...
    #we will record the CDF by recording every 1000th value in the sorted S distribution. (x8 definitions)
    #for maxima record every value
    X = df[x_col].values
    MX = df[df['global_max']==1][x_col].values
    print(df[df['global_max']==1])
    #indices = df[df['global_max']==1]["seed","frame"]    
    reg_S_reduced = []
    reg_SX = []
    reg_percentile = []
    def get_drop_index(indices=np.array([(0,0,0),(1,1,1)])):
        M = len(indices)
        output = np.zeros(M,dtype=int)
        for i in range(M):
            index = indices[i]
            seed = index[0]
            time = index[1]
            drops = np.loadtxt(args.basedir+str(seed)+"/drops.txt")[:,0].astype(int)
            output[i] = np.searchsorted(drops,time)
        return output
    dfX = df[df['global_max']==1]
    max_event = np.max( get_drop_index(dfX.index.values))
    print(max_event)
    t_reg_percentiles = []
    t_clf_percentiles = []
   
    for l in range(8):
        reg_S = reg[l].predict(X[:,:3+2*l]) 
        reg_SX.append(reg[l].predict(MX[:,:3+2*l]))
        reg_percentile.append(1.0 - stats.mannwhitneyu(reg_S, reg_SX[l])[0] / (len(reg_SX[l]) *len(reg_S)))
        reg_S_reduced.append(np.sort(reg_S)[:-1:1000])
        if l ==7:
          for t in range(max_event+1):
            print(t, np.sum(get_drop_index(dfX.index)==t))
            dfXt = dfX[get_drop_index(dfX.index) == t]
            reg_St = reg[l].predict(dfXt[x_col].values)
            t_reg_percentiles.append(1.0 - stats.mannwhitneyu(reg_S, reg_St)[0] / (len(reg_St) *len(reg_S)))
 
 
    clf_S_reduced = []
    clf_SX = []
    clf_percentile = []
    for l in range(8):
        clf_S = clf[l].decision_function(X[:,:3+2*l])
        clf_SX.append(clf[l].decision_function(MX[:,:3+2*l]))
        clf_percentile.append(1.0 - stats.mannwhitneyu(clf_S, clf_SX[l])[0] / (len(clf_SX[l]) *len(clf_S)))
        clf_S_reduced.append(np.sort(clf_S)[:-1:1000])
        if l ==7:
          for t in range(max_event+1):
            dfXt = dfX[get_drop_index(dfX.index) == t]
            clf_St = clf[l].decision_function(dfXt[x_col].values)
            t_clf_percentiles.append(1.0 - stats.mannwhitneyu(clf_S, clf_St)[0] / (len(clf_St) *len(clf_S)))
 

    
    output = {}
    for var in ["clf_S_reduced", "clf_SX", "clf_percentile", "reg_S_reduced", "reg_SX", "reg_percentile","t_clf_percentiles","t_reg_percentiles"]:
        output[var] = eval(var)

       #now a bit finer - P(R) conditioned on (O,G), and P(O,G). But we can just quote the full numbers.

    df['pair'] = df[[(1,'o'), (1,'g')]].apply(tuple, axis=1)
    frequencies = df.groupby('pair').size()
    max_frequencies = df[df["global_max"] == True].groupby('pair').size()

    for var in ["frequencies", "max_frequencies"]:
        output[var] = eval(var)
    output["basedir"] = args.basedir
    #parse the basedir to produce an output filename
    outdir = "/projects/ajliu/ridout/data/softness_percentiles/"
    os.makedirs(outdir,exist_ok=True)
    #god I hate doing this crap
    if re.search("memsshear", args.basedir) is not None:
        outfile = "memsshear_"
    else:
        outfile = "nasshear_" 
          
    outfile += str(args.dim) + "d_N="+str(args.NP)+"_"
     
    if re.search("phi=.*_", args.basedir) is not None:
        outfile += re.search("phi=[0123456789\.]*_", args.basedir)[0][:-1]
    elif re.search("Lp=.*_", args.basedir) is not None:
        outfile += re.search("Lp=[0123456789\.]*_", args.basedir)[0][:-1] 
    else:
        print("Some difficulty parsing basedir...")
        raise SystemExit

    if args.notype:
        outfile += "_notype"
    elif args.nogaps:
        outfile += "_nogaps"
    with open(outdir + outfile, "wb") as f:
        pickle.dump(output, f) 
    

main(ARGS)     


