#This calculation is structured somewhat differently.
#For each trajectory, form a training set using the complement of that trajectory.
#Compute correlation functions of mode d2min, rescaled mode d2min, Z, S1, S8

import os
import argparse
import pickle
import re

import numpy as np 
import scipy.stats as stats
import pandas as pd

import softness.regressor as regressor
import softness.classifier as classifier

import corr
import iowrappers as io

from netCDF4 import Dataset
def get_valid_seeds(basedir):
    seeds = []
    for i in range(100):
        if os.path.isfile(basedir+str(i)+"/descriptor_df.pkl"):
            seeds.append(i)

    return seeds

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


def infer_rattlermap(directory, _time, dim):

    time = str(int(_time))

    try:
        mode = np.loadtxt(directory+"lowmodesEnth/"+time+".txt")[1:]
    except:
        mode = np.loadtxt(directory+"lowmodes2/"+time+".txt")[1:]

    mode = mode.reshape((-1,dim))
    rattlermap = []
    for i in range(len(mode)):
        if not(np.all(mode[i] == np.zeros(dim))):
            rattlermap.append(i)
    return np.array(rattlermap, dtype=int)

PARSER = argparse.ArgumentParser("Load descriptor dataframes, analyze regression softness. Also compute correlation functions by brute force while we are at it.")
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
    output_dfs = []
    for seed in seeds:
        filename = args.basedir+ str(seed) + "/descriptor_df.pkl"
        with open(filename, "rb") as f:
            seed_df = pickle.load(f)
        temp_df = get_seed_complement_df(df=full_train_df, seed=seed)
        temp_output_dfs = []
        temp_output_dfs.append(get_seed_df(df=full_test_df, seed=seed)["particle_type"])
        Z = np.sort(temp_df[(1,'o')].values)
        unsorted_Z = seed_df[(1,'o')].values #for correlation functions
        max_Z = get_seed_df(df=full_test_df, seed=seed)[(1,'o')]
        def Z_percentile(X):
            return 1.0 - np.mean((np.searchsorted(Z,X,side="left"),np.searchsorted(Z,X,side="right")))/len(Z) 
        temp_output_dfs.append(max_Z.apply(Z_percentile).rename(columns={(1,"o"):"Z"}))
        temp_output_dfs.append(max_Z.rename(columns={(1,"o"):"abs_Z"}))
        reg_1 = None
        reg_8 = None
        for l in [1,2,3,4,5,6,7,8]:
           reg = regressor.get_regressor(temp_df, x_col[:1+2*l], 'D2min_normed')
           clf = classifier.get_classifier(temp_df[temp_df['local'] != 0], x_col[:1+2*l], 'local', clf_type="SV")
           test_descriptor_df = get_seed_df(df=full_test_df, seed=seed)[x_col[:1+2*l]]
              
           #compute distributions to generate percentiles
           X = temp_df[x_col[:1+2*l]].values
           reg_S = np.sort(reg.predict(X))
           clf_S = np.sort(clf.decision_function(X))
           M = len(X)
           def reg_percentile(X):
               return np.mean((np.searchsorted(reg_S,reg.predict(X),side="left"),np.searchsorted(reg_S,reg.predict(X),side="right"))) / M

           def clf_percentile(X):
               return np.mean((np.searchsorted(clf_S,clf.decision_function(X),side="left"),np.searchsorted(clf_S,clf.decision_function(X),side="right"))) / M
         
           
           
           temp_output_dfs.append(generate_output_df(desc_df=test_descriptor_df, 
               output_funcs = (reg_percentile, clf_percentile), names = ("reg_"+str(l), "clf_"+str(l))))

           #for correlation functions need unsorted stuff
           if l == 1:
             reg_1 = reg
           if l == 8:
             reg_8 = reg

        temp_output_dfs = pd.concat(temp_output_dfs, axis=1).rename(columns={0:"Z"})
        output_dfs.append(temp_output_dfs)
        try:
            drops = np.loadtxt(args.basedir+str(seed)+"/drops.txt")[:,0].astype(int) 
        except:
            drops = np.array(np.loadtxt(args.basedir+str(seed)+"/drops.txt")[0].astype(int))
        #correlations
        try:
            frames = np.arange(len(drops),dtype=int)
        except:
            frames = np.array([0],dtype=int)
        for frame in frames:
            try:
                time =drops[frame]
            except:#only one drop...
                time = drops 
            print("getting drop indices...")
            #frame_args = get_drop_index(seed_df.index.values).astype(int) == frame
            print(np.asarray([[x for x in t] for t in seed_df.index.values], dtype=int))
            #print((np.array(seed_df.index.values,dtype=int)[:,1]))
            frame_args = np.asarray([[x for x in t] for t in seed_df.index.values], dtype=int)[:,1] ==int(time)
    
            print(len(seed_df[frame_args]))
            X = seed_df[frame_args][x_col].values
            X1 = X[:,:3]
            Z = X[:,2]
            print("frame", frame, "time", time)
            #nonrattlers = np.array(np.loadtxt(args.basedir+str(seed)+"/rattlermaps/"+str(frame)+".txt")).astype(int)
            nonrattlers = infer_rattlermap(args.basedir+str(seed)+"/", time, args.dim)
            if len(seed_df[frame_args]) == len(nonrattlers):
                print(len(nonrattlers), len(seed_df[frame_args]["D2min"].values))
                fields = np.zeros((7,len(nonrattlers)))
                fields[0,:] = seed_df[frame_args]["D2min"].values
                fields[1,:] = seed_df[frame_args]["D2min_normed"].values
                fields[2,:] = Z
                fields[3,:] = reg_1.predict(X1)
                fields[4,:] = reg_8.predict(X)
                fields[5,:] = seed_df[frame_args]["D2min1"].values
                fields[6,:] = seed_df[frame_args]["D2min_normed1"].values
                fields = fields.transpose()
                statefilename = args.basedir + str(seed)+"/state.nc" 
                state = Dataset(statefilename,"r")
                pos = state.variables['pos'][time].reshape((args.NP,args.dim)) #virtual
                rad = state.variables['rad'][time]
                pos = pos[nonrattlers,:]
                rad = rad[nonrattlers]
                classes = (rad == np.max(rad))
                box_mat = np.array(state.variables['BoxMatrix'][time].reshape((args.dim, args.dim))).T
                pair_counts, prod_sum, avg_sum = corr.compute_brute_correlations_classes(pos=pos, box=box_mat, r_cut = 0.5*box_mat[0,0], virtual=True, fields = fields, bin_spacing = 0.1, classes = classes)
                print("self counts", pair_counts[0,:])
                key = [(seed, frame)]
                output = {"frame":key }
                output["pair_counts"] = [pair_counts]
                output["D2min_prod_sum"] = [prod_sum[:,0]]
                output["D2min_avg_sum"] = [avg_sum[:,0]]
                output["D2min_normed_prod_sum"] = [prod_sum[:,1]]
                output["D2min_normed_avg_sum"] = [avg_sum[:,1]]
                output["Z_prod_sum"] = [prod_sum[:,2]]
                output["Z_avg_sum"] = [avg_sum[:,2]]
                output["S1_prod_sum"] = [prod_sum[:,3]]
                output["S1_avg_sum"] = [avg_sum[:,3]]
                output["S8_prod_sum"] = [prod_sum[:,4]]
                output["S8_avg_sum"] = [avg_sum[:,4]]
                output["D2min1_prod_sum"] = [prod_sum[:,5]]
                output["D2min1_avg_sum"] = [avg_sum[:,5]]
                output["D2min1_normed_prod_sum"] = [prod_sum[:,6]]
                output["D2min1_normed_avg_sum"] = [avg_sum[:,6]]

                df = pd.DataFrame(output)
                df.set_index("frame", inplace=True)
                os.makedirs(args.basedir+str(seed)+"/corrfuncs/", exist_ok=True)
                outfilename = args.basedir+str(seed)+"/corrfuncs/"+str(int(time))+".pkl"
                print(outfilename)
                with open(outfilename,"wb") as f:
                    pickle.dump(df, f)
            else:
                print("rattlers?", X.shape,len(nonrattlers))
                continue
            print("checkpoint", frame)
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
    output = {"all_percentiles": output_dfs, "traj_mean_percentiles": traj_mean_list, "summary_stats": summary_stats, "Z_counts": Z_counts}
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
