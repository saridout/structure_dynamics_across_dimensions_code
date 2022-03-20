import pickle
import argparse
import json
import re
import os

from timeit import default_timer as timer

import numpy as np
import pandas as pd

import gsd
import gsd.fl
import netCDF4

import phom
import softness as soft

PARSER = argparse.ArgumentParser("Create descriptor dataframe.")
PARSER.add_argument(
    "-n", "--NP", help="Number of particles in system.", type=int)
PARSER.add_argument("-s", "--seed", help="Trajectory seed.", type=int)
PARSER.add_argument("-f", "--frequency",
                    help="Event sampling frequency.", type=int)
PARSER.add_argument(
    "-w", "--window", help="Window, in number of events, for hard D2min. Default: 1, i.e. don't record.", type=int, default=1)
PARSER.add_argument('--no_old', help="Suppress old structure functions", action="store_true")
PARSER.add_argument('--dim', help="Spatial dimension.", type=int, default=2)
PARSER.add_argument('--basedir', help = "/blah/blah/blah= , seed goes after = ")
ARGS = PARSER.parse_args()
DIM = ARGS.dim

print(ARGS)

def main(args):

    if not args.basedir == None:
        folder = args.basedir + str(args.seed)+"/"
    else:
        folder = "/home1/ridout/states/memsshear/2d/2dshear_N=16384_phi=0.95_dγ=0.0001_rP=500_seed=" + \
        str(args.seed)+"/"
    try:
        deco_filename = folder+"dropdeco3_N="+str(args.NP)+"_dropthresh=perc.gsd"
    except:
        with open(folder+"params.json", "r") as f:
           params = json.load(f)
        deco_filename = re.sub("/projects/jamming","/home1", params["filename_dropdeco"])
    state_filename = folder+"state.nc"
    
    state = netCDF4.Dataset(state_filename, 'r')
    full_df = []
    jason_strf = {}
    old_strf = {} 
    ang = {}
    D2min_final1 = {}
    D2min_final2 = {}

    final_rattlers = {}
    classes = {}
    D2min_normed1 = {}
    D2min_normed2 = {}
    with gsd.fl.GSDFile(name=deco_filename, mode='rb') as decoFlow:
        M = decoFlow.nframes
        print("M:", M)
        if args.no_old:
            output_filename = folder + "descriptor_df.pkl"
        else:
            output_filename = folder + "descriptor_df_jason.pkl"
        #check if the data already exist
        data_size = M
        existing_data_size = 0
        if os.path.isfile(output_filename):
           with open(output_filename, "rb") as f:
              existing_output = pickle.load(f)
              existing_data_size = len(existing_output.groupby("time"))
              #full_df.append(existing_output)
        print("data size vs existing:", data_size, existing_data_size, flush=True)
        #if data_size == existing_data_size: 
            #raise SystemExit
        print("Recalculating")
        existing_data_size = 0
        if args.window > 1:
            d2min_list2 = np.zeros((args.window, args.NP))
            rescaled_list2 = np.copy(d2min_list2)
            d2min_list1 = np.zeros((args.window, args.NP))
            rescaled_list1 = np.copy(d2min_list1)
        for event in range(existing_data_size, M):
            # always compute D2min, whether or not event % frequency == 0
            time = decoFlow.read_chunk(frame=event, name='tc')[0]

            print(M, time)
            try:
                mode = np.array(decoFlow.read_chunk(
                    frame=event, name='rawModeH'), float)
            except:
                mode = np.array(decoFlow.read_chunk(
                    frame=event, name='rawMode'), float)
            start = timer()
            (embed, rad2) = soft.tri.get_configuration(state, time)
            end = timer() 
            print("Get configuration time:",end - start)
            start = timer()
            comp = soft.tri.construct_triangulation(embed, rad2)
            end = timer()
            print("Triangulation construction time:",end-start)
            print("Cells:",comp.ndcells)
            (rattlers, comp, embed, rad2) = soft.tri.remove_rattlers(
                comp, embed, rad2)

            new_mode = []
            rattlers = set(rattlers)
            for p in range(len(mode) // DIM):
                if p not in rattlers:
                    new_mode.extend(mode[DIM*p:DIM*p+DIM])

            mode = np.array(new_mode)
            D2min2 = soft.tri.calc_D2min(comp, mode, embed, layers=2)
            D2min1 = soft.tri.calc_D2min(comp, mode, embed, layers=1)
            rescaled2 = soft.tri.normalize_D2min( D2min2, comp, embed, layers=2)
            rescaled1 = soft.tri.normalize_D2min( D2min1, comp, embed, layers=1)

            minima2, maxima2 = soft.tri.find_local_extrema(D2min2, comp,layers=2)
            minima1, maxima1 = soft.tri.find_local_extrema(D2min1, comp,layers=1)

            local2 = []
            for p in range(len(mode) // DIM): 
                if p in minima2:
                    local2.append(-1)
                elif p in maxima2:
                    local2.append(1) 
                else:
                    local2.append(0)

            local1= []
            for p in range(len(mode) // DIM): 
                if p in minima1:
                    local1.append(-1)
                elif p in maxima2:
                    local1.append(1) 
                else:
                    local1.append(0)
            # insert logic for back-updating "window activity". take care with rattler map here.
            all_D2min2 = np.zeros(args.NP)
            all_rescaled2 = np.zeros(args.NP)
            all_D2min1 = np.zeros(args.NP)
            all_rescaled1 = np.zeros(args.NP)
 
            ind = 0
            for p in range(args.NP):
                if p in rattlers:
                    all_D2min2[p] = np.inf
                    all_rescaled2[p] = np.inf
                    all_D2min1[p] = np.inf
                    all_rescaled1[p] = np.inf
                else:
                    all_D2min2[p] = D2min2[ind]
                    all_rescaled2[p] = rescaled2[ind] 
                    all_D2min1[p] = D2min1[ind]
                    all_rescaled1[p] = rescaled1[ind] 
                    ind += 1
            if event < ARGS.window and ARGS.window > 1:
                d2min_list2[event] = all_D2min2
                rescaled_list2[event] = all_rescaled2
                d2min_list1[event] = all_D2min1
                rescaled_list1[event] = all_rescaled1                
            elif ARGS.window > 1:
                for t in range(args.window-1):
                    d2min_list2[t] = d2min_list2[t+1]
                    rescaled_list2[t] = rescaled_list2[t+1]
                    d2min_list1[t] = d2min_list1[t+1]
                    rescaled_list1[t] = rescaled_list1[t+1]
                d2min_list2[args.window-1] = all_D2min2
                rescaled_list2[args.window-1] = all_rescaled2
                d2min_list1[args.window-1] = all_D2min1
                rescaled_list1[args.window-1] = all_rescaled1
            # now check if we have a need to compute structure functions
            particles = np.array(range(args.NP - len(rattlers)))
            print( len(rattlers), len(rad2))
            if event % args.frequency == 0:
                D2min_final2[event] = D2min2
                D2min_final1[event] = D2min1

                final_rattlers[event] = rattlers
                D2min_normed2[event] = soft.tri.normalize_D2min(
                    D2min2, comp, embed, layers=2)
                D2min_normed1[event] = soft.tri.normalize_D2min(
                    D2min1, comp, embed, layers=1)

                start = timer()
                jason_strf[event] = soft.strfunc.get_go_counts(
                    particles, comp, embed, rad2, 8)
                end = timer()
                print("Gap/overlap time:",end-start)
                rel_left = np.arange(0.8, 2.0, 0.05)
                rel_width = 0.05
                AA_d = 5/6
                AB_d = 1.0
                BB_d = 7/6
                AA_left = rel_left*AA_d
                AB_left = rel_left*AB_d
                BB_left = rel_left*BB_d
                AA_width = rel_width*AA_d
                AB_width = rel_width*AB_d
                BB_width = rel_width*BB_d
                classes[event] = (rad2 == np.max(rad2)).astype(int)
                particles_A = np.where(np.logical_not(classes[event]))[0]
                particles_B = np.where(classes[event])[0]
                print(len(particles_A), len(particles_B))
                radial_params_A = [{"mean": AA_left+AA_width/2, "sigma": np.full(len(rel_left), AA_width/2.0)}, {
                    "mean": AB_left+AB_width/2.0, "sigma": np.full(len(rel_left), AB_width/2.0)}]
                radial_params_B = [{"mean": AB_left+AB_width/2, "sigma": np.full(len(rel_left), AB_width/2.0)}, {
                    "mean": BB_left+BB_width/2.0, "sigma": np.full(len(rel_left), BB_width/2.0)}]
                if not args.no_old:
                    start = timer()
                    df_A = soft.strfunc.triangulation_radial_func(
                        embed, comp, interesting_particles=particles_A, classes=classes[event], func=soft.strfunc.gaussian_sf, params=radial_params_A)
                    df_B = soft.strfunc.triangulation_radial_func(
                        embed, comp, interesting_particles=particles_B, classes=classes[event], func=soft.strfunc.gaussian_sf, params=radial_params_B)
                    end = timer()
                    print("Old radial structure function time:",end-start)
                    df_A = df_A.sort_values("particle")
                    df_B = df_B.sort_values("particle")
                    old_strf[event] = pd.concat((df_A, df_B))
                jason_strf[event] = jason_strf[event].sort_values("particle")
                if not args.no_old:
                #now angular...many more options, so work hard.
                    rel_R = np.array([2.554,]*4 + [1.648, ]*2 + [1.204, ]*4 + [0.933, ]*4 + [0.695,]*4) 
                    zeta = np.array([ 1, 1, 2, 2, 1, 2, 1, 2, 4, 16, 1, 2, 4, 16, 1, 2, 4, 16])
                    lamb = np.array([ -1, 1, -1, 1] + [1, ]*14)               
                
                    angular_params = [{"R":rel_R*AA_d, "zeta":zeta, "lamb": lamb}, ]*3 
                    start = timer()
                    df_A_ang = soft.strfunc.triangulation_angular_func(
                        embed, comp, interesting_particles=particles_A, classes=classes[event], func=soft.strfunc.bp_angular_sf, params=angular_params, key_prefix="A")
                    df_B_ang = soft.strfunc.triangulation_angular_func(
                        embed, comp, interesting_particles=particles_B, classes=classes[event], func=soft.strfunc.bp_angular_sf, params=angular_params, key_prefix="B")
                    end = timer()
                    print("Angular structure function time:", end - start)
                    df_A_ang = df_A_ang.sort_values("particle")
                    df_B_ang = df_B_ang.sort_values("particle")
                    ang[event] = pd.concat((df_A_ang, df_B_ang))
                    ang[event].fillna(0, inplace=True)
            if (event - args.window + 1) >= 0 and (event - args.window+1) % args.frequency == 0:  # record
                if args.window > 1:
                    window_max_all = np.max(d2min_list2, axis=0)
                    window_max_rescaled = np.max(rescaled_list2,axis=0)
                    window_max = []
                    rescaled_max = []
                    for p in range(args.NP):
                        if p not in final_rattlers[event - args.window+1]:
                            window_max.append(window_max_all[p])
                            rescaled_max.append(window_max_rescaled[p])

                # combine structure dfs
                if args.no_old:
                    df = jason_strf[event - args.window+1]
                else: 
                    df = pd.merge(jason_strf[event - args.window+1], old_strf[event - args.window+1], on="particle")
                    df = pd.merge(df, ang[event - args.window+1], on="particle")
                    del old_strf[event - args.window+1]
                    del ang[event - args.window+1]
                del jason_strf[event - args.window+1]
                print(df)
                print(len(df))
                print(len(D2min_final2[event - args.window+1]))
                df.insert(0, "D2min", D2min_final2[event - args.window+1])
                df.insert(0, "D2min1", D2min_final1[event - args.window+1])
                df.insert(1, "D2min_normed", D2min_normed2[event - args.window+1])
                df.insert(1, "D2min_normed1", D2min_normed1[event - args.window+1])

                if args.window > 1:
                    df.insert(2, "D2min_window_max", window_max)
                    df.insert(2,"rescaled_window_max",rescaled_max)
                else:
                    df.insert(2, "local", local2)
                    df.insert(2, "local1", local1)

                df.insert(3, "global_max", D2min_final2[event - args.window+1] == np.max(D2min_final2[event - args.window+1]))
                df.insert(4, "particle_class", classes[event - args.window+1])
                df.insert(0, "seed", args.seed)
                df.insert(1, "time", time)
                df.insert(2, "frame", event - args.window+1) 
                df.set_index(['seed', 'time', 'particle'], inplace=True)
                full_df.append(df)

    full_df = pd.concat(full_df)

    # output
    with open(output_filename, "wb") as f:
        pickle.dump(full_df, f)

main(ARGS)
