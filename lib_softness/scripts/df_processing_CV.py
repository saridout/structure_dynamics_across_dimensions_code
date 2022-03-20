#This calculation is structured somewhat differently.
#For each trajectory, form a training set using the complement of that trajectory.

import os
import argparse
import pickle
import re

import numpy as np 
import scipy.stats as stats
import pandas as pd

import softness.regressor as regressor
import softness.classifier as classifier


def get_valid_seeds(basedir):
    seeds = []
    for i in range(100):
        if os.path.isfile(basedir+str(i)+"/descriptor_df.pkl"):
            seeds.append(i)

    return seeds

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

def get_seed_index(indices=np.array([(0,0,0),(1,1,1)])):
        M = len(indices)
        output = np.zeros(M,dtype=int)
        for i in range(M):
            index = indices[i]
            seed = int(index[0])
            output[i] = seed
        return output

def get_seed_df(df=pd.DataFrame(),seed=1):
    return df[get_seed_index(df.index) == seed ]

def get_seed_complement_df(df=pd.DataFrame(),seed=1):
    return df[get_seed_index(df.index) != seed ]

def generate_output_df(desc_df=pd.DataFrame(),output_funcs=(), names=()):
    dfs = []
    for i in range(len(output_funcs)):
        f = output_funcs[i]
        dfs.append(desc_df.apply(lambda x: pd.Series({names[i]:f(x.values.reshape(1, -1))}), axis=1))
    return pd.concat(dfs,axis=1)

PARSER = argparse.ArgumentParser("Load descriptor dataframes, analyze regression softness.")
PARSER.add_argument(
    "-n", "--NP", help="Number of particles in system.", type=int)
PARSER.add_argument('--dim', help="Spatial dimension.", type=int, default=2)
PARSER.add_argument('--basedir', help = "/blah/blah/blah= , seed goes after = ")
PARSER.add_argument('--nogaps', action="store_true", help = "Do not use gaps in descriptors.")
PARSER.add_argument('--notype', action="store_true", help = "Do not use particle type in descriptors.")
PARSER.add_argument('--maxsample', type=int, default=50000, help = "Maximum sample of particles per trajectory.")
PARSER.add_argument('--test', action="store_true", help = "Use only three trajectories to reduce runtime during testing")
ARGS = PARSER.parse_args()

def main(args):

    seeds = get_valid_seeds(args.basedir)
    if args.test:
        seeds = seeds[:3]
    #construct training set
    full_train_df = pd.DataFrame()
    full_test_df = pd.DataFrame()
    for seed in seeds:
        filename = args.basedir+ str(seed) + "/descriptor_df.pkl"
        with open(filename, "rb") as f:
            temp = pickle.load(f)
        sample = np.min((len(temp), args.maxsample))
        full_train_df = pd.concat([full_train_df, temp.sample(n=sample)])
        full_test_df = pd.concat([full_test_df, temp[temp["global_max"]==1]])


    #now for each seed, train the classifier + regressor, and compute the test statistics. 
    x_col = ['particle_type'] + [(i, t) for i in range(1,9) for t in ["g","o"] ]
    output = {}
    for rs in [""]:
      try:
        output_dfs = []
        R2 = []
        for seed in seeds:
            print(seed, rs)
            temp_df = get_seed_complement_df(df=full_train_df, seed=seed)
            temp_output_dfs = []
            temp_output_dfs.append(get_seed_df(df=full_test_df, seed=seed)["particle_type"])
            Z = np.sort(temp_df[(1,'o')].values)
            max_Z = get_seed_df(df=full_test_df, seed=seed)[(1,'o')]
            def Z_percentile(X):
                return 1.0 - np.mean((np.searchsorted(Z,X,side="left"),np.searchsorted(Z,X,side="right")))/len(Z) 
            temp_output_dfs.append(max_Z.apply(Z_percentile).rename(columns={(1,"o"):"Z"}))
            temp_output_dfs.append(max_Z.rename(columns={(1,"o"):"abs_Z"}))
            print("X NaN:", np.where(np.isnan(temp_df[x_col].values)))
            print("X inf:", np.where(np.isinf(temp_df[x_col].values)))
            print("D2min nan:", np.where(np.isnan(temp_df["D2min_normed"+rs].values)))
            print("D2min inf:", np.where(np.isinf(temp_df["D2min_normed"+rs].values)))
            for l in [1,2,3,4,5,6,7,8]:
                reg = regressor.get_regressor(temp_df, x_col[:1+2*l], 'D2min_normed'+rs)
                clf = classifier.get_classifier(temp_df[temp_df['local'+rs] != 0], x_col[:1+2*l], 'local'+rs, clf_type="SV")
                test_descriptor_df = get_seed_df(df=full_test_df, seed=seed)[x_col[:1+2*l]]
                if l == 1:
                    print("coeffs:", reg.predict(np.array(((1,0,0))).reshape(1,-1)) - reg.predict(np.array(((1,0,0))).reshape(1,-1)), reg.predict(np.array(((0,1,0))).reshape(1,-1)) - reg.predict(np.array(((0,0,0))).reshape(1,-1)) , reg.predict(np.array(((0,0,1))).reshape(1,-1)) - reg.predict(np.array(((0,0,0))).reshape(1,-1)))
                #compute distributions to generate percentiles
                X = temp_df[x_col[:1+2*l]].values
                reg_S = np.sort(reg.predict(X))
                clf_S = np.sort(clf.decision_function(X))
                M = len(X)
                def reg_percentile(X):
                    return np.mean((np.searchsorted(reg_S,reg.predict(X),side="left"),np.searchsorted(reg_S,reg.predict(X),side="right"))) / M

                def clf_percentile(X):
                    return np.mean((np.searchsorted(clf_S,clf.decision_function(X),side="left"),np.searchsorted(clf_S,clf.decision_function(X),side="right"))) / M
                
                if l == 8:
                    normed = temp_df["D2min_normed"+rs]
                    X = temp_df[x_col[:1+2*l]].values
                    mean = np.mean(normed)
                    R2.append( 1 -  np.sum((normed - reg.predict(X))**2)/ np.sum((normed - mean)**2)) 
                
                temp_output_dfs.append(generate_output_df(desc_df=test_descriptor_df, 
                    output_funcs = (reg_percentile, clf_percentile), names = ("reg_"+str(l), "clf_"+str(l))))
            temp_output_dfs = pd.concat(temp_output_dfs, axis=1).rename(columns={0:"Z"})
            print(temp_output_dfs)
            output_dfs.append(temp_output_dfs)

        output_dfs = pd.concat(output_dfs)
        #now we can go ahead and compute some summary statistics.
        traj_mean_list = output_dfs.groupby("seed").mean()
        summary_stats = {}
        funcs = [np.mean, lambda x : np.std(x) / np.sqrt(len(seeds))] 
        names = ["mean", "stderr"]
        for i in range(len(names)):
            summary_stats[names[i]] = traj_mean_list.apply(funcs[i], axis=0)
        summary_stats = pd.DataFrame(summary_stats)
        #Z_counts = full_train_df.groupby("particle_type")[(1,'o')].value_counts()
        Z_counts = full_train_df.groupby(['particle_type',(1,'o')]).size().unstack(fill_value=0)
        print(Z_counts)
        output.update({ "all_percentiles"+rs: output_dfs, "traj_mean_percentiles"+rs: traj_mean_list, "summary_stats"+rs: summary_stats, "Z_counts": Z_counts, "R2"+rs: R2})
        print("R2"+rs, R2, "?")
      except Exception as e:
        print(e)

    if not output:
       print("no output!")
       raise SystemExit
    output["basedir"] = args.basedir
    #parse the basedir to produce an output filename
    outdir = "/home1/ridout/data/softness_percentiles_CV/"
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
    if re.search("T=.*\/", args.basedir) is not None:
        outfile += re.search("T=[0123456789e\-\.]*\/", args.basedir)[0][:-1]
    if args.notype:
        outfile += "_notype"
    elif args.nogaps:
        outfile += "_nogaps"
    with open(outdir + outfile, "wb") as f:
        pickle.dump(output, f) 
    

  
main(ARGS)
