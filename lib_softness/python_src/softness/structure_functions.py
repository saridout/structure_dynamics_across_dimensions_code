import sys, os
sys.path.insert(0, '../../lib_persistent_homology/')
sys.path.insert(0, '../../lib_persistent_homology/python_src/')

import numpy as np
import scipy as sp
import pandas as pd

import phom
from numba import jit

def get_go_counts(particles, comp, embed, rad2, max_tri_dist, overlap_cutoffs=np.zeros(3)):


    r2norm = np.min(rad2)

    vert_types = np.where(rad2==r2norm, 'A', 'B')
    edge_vert_types = [vert_types[comp.get_facets(c)] for c in range(*comp.dcell_range[1])]
    edge_verts = [comp.get_facets(c) for c in range(*comp.dcell_range[1])]
    map_to_int = lambda c : int(ord(c) - ord('A'))
    edge_particle_pair_types = [map_to_int(x[0]) + map_to_int(x[1]) for x in edge_vert_types]
    #edge_types = np.where(alpha_vals[comp.dcell_range[1][0]:comp.dcell_range[1][1]] < overlap_alpha_cutoffs[edge_paricle_pair_types], 'o', 'g')
    #lazy handling of particle radii for now...
    cutoffs = overlap_cutoffs + np.array([5/6, 1.0, 7/6])
    def dist(i, j):
        return np.linalg.norm(embed.get_diff(embed.get_vpos(i),embed.get_vpos(j)))
    edge_types = np.where( [dist(x[0],x[1]) for x in edge_verts] < cutoffs[edge_particle_pair_types], 'o', 'g')
    edge_counts = phom.calc_radial_edge_counts(particles, edge_types, comp, max_tri_dist)


    col_set = set()

    for p in edge_counts:
        for rad_dist in edge_counts[p]:
            for edge_type in edge_counts[p][rad_dist]:
                col_set.add((rad_dist, edge_type))

    col_index = {x:i for i, x in enumerate(sorted(col_set))}

    s_list = []

    for p in edge_counts:



        s = [0]*len(col_index)

        for rad_dist in edge_counts[p]:
            for edge_type in edge_counts[p][rad_dist]:
                s[col_index[(rad_dist, edge_type)]] =  edge_counts[p][rad_dist][edge_type]

        ptype = vert_types[p]

        if ptype == 'A':
            ptype = 0
        else:
            ptype = 1

        s_list.append([p, ptype]+s)

    columns = ['particle', 'particle_type'] + sorted(col_set)

    df = pd.DataFrame(s_list, columns=columns)

    return df


def get_edge_type_counts(particles, comp, embed, rad2, max_tri_dist):

    r2norm = np.min(rad2)

    if embed.dim == 2:
        alpha_vals = phom.calc_alpha_vals_2D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 3:
        alpha_vals = phom.calc_alpha_vals_3D(comp, embed, rad2, alpha0=-r2norm)

    vert_types = np.where(rad2==r2norm, 'A', 'B')

    edge_types = []
    for c in range(*comp.dcell_range[1]):

        verts = comp.get_facets(c)

        if alpha_vals[c] < 0.0:
            label = 'o'
        else:
            label = 'g'

        label += ''.join(sorted(vert_types[verts]))

        edge_types.append(label)

    edge_counts = phom.calc_radial_edge_counts(particles, edge_types, comp, max_tri_dist)

    col_set = {(1, 'dg'), (1, 'do')}
    for i in range(1, max_tri_dist+1):
        col_set.add((i, 'g'))
        col_set.add((i, 'o'))

    for i in range(2, max_tri_dist+1):
        col_set.add((i, 'dgA'))
        col_set.add((i, 'dgB'))
        col_set.add((i, 'doA'))
        col_set.add((i, 'doB'))

    col_index = {x:i for i, x in enumerate(sorted(col_set))}

    s_list = []

    for p in edge_counts:

        s = [0]*len(col_index)

        ptype = vert_types[p]

        for rad_dist in edge_counts[p]:
            gAA = edge_counts[p][rad_dist].get('gAA', 0)
            gAB = edge_counts[p][rad_dist].get('gAB', 0)
            gBB = edge_counts[p][rad_dist].get('gBB', 0)

            oAA = edge_counts[p][rad_dist].get('oAA', 0)
            oAB = edge_counts[p][rad_dist].get('oAB', 0)
            oBB = edge_counts[p][rad_dist].get('oBB', 0)

            s[col_index[(rad_dist, 'g')]] = gAA + gAB + gBB
            s[col_index[(rad_dist, 'o')]] = oAA + oAB + oBB

            if rad_dist == 1:
                if ptype =='A':
                    s[col_index[(rad_dist, 'dg')]] = gAA - gAB
                    s[col_index[(rad_dist, 'do')]] = oAA - oAB
                else:
                    s[col_index[(rad_dist, 'dg')]] = gAB - gBB
                    s[col_index[(rad_dist, 'do')]] = oAB - oBB

            else:
                s[col_index[(rad_dist, 'dgA')]] = gAA - gAB
                s[col_index[(rad_dist, 'dgB')]] = gBB - gAB

                s[col_index[(rad_dist, 'doA')]] = oAA - oAB
                s[col_index[(rad_dist, 'doB')]] = oBB - oAB



        if ptype == 'A':
            ptype = 0
        else:
            ptype = 1

        s_list.append([p, ptype]+s)

    columns = ['particle', 'particle_type'] + sorted(col_set)

    df = pd.DataFrame(s_list, columns=columns)

    return df



def get_tri_phenotype_counts(particles, comp, embed, rad2, max_tri_dist):


    r2norm = np.min(rad2)

    if embed.dim == 2:
        alpha_vals = phom.calc_alpha_vals_2D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 3:
        alpha_vals = phom.calc_alpha_vals_3D(comp, embed, rad2, alpha0=-r2norm)

    tri_dist = {}

    for p in particles:
        tri_dist[p] = {}
        for tri in comp.get_cofaces(p, 2):

            verts = list(comp.get_faces(tri, 0))
            verts.remove(p)

            edges = list(comp.get_faces(tri, 1))

            vphen = np.where(rad2[verts] == r2norm, '0', '1')
            vlabel = 'p' + ''.join(sorted(vphen))

            ephen = []
            for ei in edges:

                if alpha_vals[ei] > 0.0:

                    everts = comp.get_facets(ei)
                    if p in everts:
                        everts.remove(p)
                        elabel = 'p'
                    else:
                        elabel = ''

                    elabel = elabel + ''.join(sorted(np.where(rad2[everts] == r2norm, '0', '1')))

                    ephen.append(elabel)

            glabel = ''.join(sorted(ephen))

            tri_dist[p][(vlabel, glabel)] = tri_dist[p].get((vlabel, glabel), 0) + 1



    edge_types = np.where(alpha_vals[comp.dcell_range[1][0]:comp.dcell_range[1][1]] < 0.0, 'o', 'g')

    edge_counts = phom.calc_radial_edge_counts(particles, edge_types, comp, 1)


    col_set = set()

    for p in tri_dist:
        for tri_type in tri_dist[p]:
            col_set.add(tri_type)


    col_index = {x:i for i, x in enumerate(sorted(col_set))}

    s_list = []

    for p in tri_dist:

        s = [0]*len(col_index)

        for tri_type in tri_dist[p]:
            s[col_index[tri_type]] =  tri_dist[p][tri_type]

        gaps = edge_counts[p][1].get('g', 0)
        overlaps = edge_counts[p][1].get('o', 0)

        s_list.append([p, 0 if rad2[p]==r2norm else 1, gaps, overlaps]+s)

    columns = ['particle', 'particle_type', 'g', 'o'] + sorted(col_set)

    df = pd.DataFrame(s_list, columns=columns)

    return df


def get_tri_counts(particles, comp, embed, rad2, max_tri_dist):


    r2norm = np.min(rad2)

    if embed.dim == 2:
        alpha_vals = phom.calc_alpha_vals_2D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 3:
        alpha_vals = phom.calc_alpha_vals_3D(comp, embed, rad2, alpha0=-r2norm)

    tri_dist = phom.calc_radial_tri_distribution(particles, alpha_vals, comp, max_dist=max_tri_dist)


    col_set = set()

    for p in tri_dist:
        for rad_dist in tri_dist[p]:
            for tri_type in tri_dist[p][rad_dist]:
                col_set.add((rad_dist, tri_type))


    col_index = {x:i for i, x in enumerate(sorted(col_set))}

    s_list = []

    for p in tri_dist:

        s = [0]*len(col_index)

        for rad_dist in tri_dist[p]:
            for tri_type in tri_dist[p][rad_dist]:
                s[col_index[(rad_dist, tri_type)]] =  tri_dist[p][rad_dist][tri_type]

        s_list.append([p]+s)

    columns = ['particle'] + sorted(col_set)

    df = pd.DataFrame(s_list, columns=columns)

    return df


def get_cycle_counts(particles, comp, embed, rad2, max_tri_dist):

    r2norm = np.min(rad2)

    if embed.dim == 2:
        alpha_vals = phom.calc_alpha_vals_2D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 3:
        alpha_vals = phom.calc_alpha_vals_3D(comp, embed, rad2, alpha0=-r2norm)

    simp_comp = phom.join_dtriangles_2D(comp, alpha_vals)

    cycle_dist = {}
    for p in particles:

        cycle_dist[p] = {0: {}}

        cofaces = simp_comp.get_cofaces(p, embed.dim)

        for cf in cofaces:
            size = len(simp_comp.get_facets(cf))

            cycle_dist[p][0][size] = cycle_dist[p][0].get(size, 0) + 1



    col_set = set()

    for p in cycle_dist:
        for rad_dist in cycle_dist[p]:
            for size in cycle_dist[p][rad_dist]:
                col_set.add((rad_dist, size))

    col_index = {x:i for i, x in enumerate(sorted(col_set))}

    s_list = []

    for p in cycle_dist:

        s = [0]*len(col_index)

        for rad_dist in cycle_dist[p]:
            for size in cycle_dist[p][rad_dist]:
                s[col_index[(rad_dist, size)]] =  cycle_dist[p][rad_dist][size]


        s_list.append([p]+s)

    columns = ['particle'] + sorted(col_set)

    df = pd.DataFrame(s_list, columns=columns)

    return df

def get_gap_angle_classes(particles, comp,embed, rad2,num_angles=128, verbose=False):
    r2norm = np.min(rad2)
    if embed.dim == 2:
        alpha_vals =phom.calc_alpha_vals_2D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 3:
        alpha_vals = phom.calc_alpha_vals_3D(comp, embed, rad2, alpha0=-r2norm)


    classes = phom.calc_gap_angle_class2D(particles, alpha_vals, comp, embed,num_angles=num_angles, verbose=verbose)


    col_set = set()

    col_set.add("class")

    s_list = []
    for p in particles:
            s_list.append([p, classes[p]])

    columns = ['particle'] + sorted(col_set)

    df = pd.DataFrame(s_list, columns=columns)

    return df

#intended usage: x is a single distance, mean and sigma are arrays of length 1xN_descriptors
#@jit
def gaussian_sf(x,mean=1.0,sigma=0.5):
    return np.exp(-(x-mean)**2/(2*sigma**2))

def squarebin_sf(x,left=1.0,width=0.5):
    return np.logical_and(x> left, x < left+width)

#one class (e.g. neighbours with particle type A)
#@jit
def broadcast_radial_func(neighb_dist,func=gaussian_sf,params={"mean":0.5*np.arange(2,11),"sigma":np.full(9,0.5)}):
  output = np.zeros(len(params[list(params.keys())[0]]))#access an arbitrary element of dictionary
  for r in neighb_dist:
    output += func(r,**params)
  return output
#one test particle
#all classes of neighbour. now params is (K*M), where K is number of classes and M is number of descriptors per pair
#params should be a list of dicts
#@jit
def all_classes_radial_func(neighb_list,neighb_dist,classes=np.zeros(4096),func=gaussian_sf,params=[{"mean":0.25*np.arange(4,22),"sigma":np.full(18,0.25)}, {"mean":0.35*np.arange(4,22),"sigma":np.full(18,0.35)}]):
  N_classes = len(params)
  output = np.zeros((N_classes,len(params[0][list(params[0].keys())[0]])))
  for c in range(N_classes):
    output[c] += broadcast_radial_func(neighb_dist[np.where(classes[neighb_list]==c)[0]],func=func,params=params[c])

  return output
#now here's one way to put it all together
def triangulation_radial_func(embed,comp,interesting_particles=np.arange(2048),classes=np.concatenate((np.zeros(2048,dtype=int),np.full(2048,1,dtype=int))),func=gaussian_sf,params=[{"mean":(5/96)*np.arange(6,40),"sigma":np.full(34,5/96)},{"mean":(3/48)*np.arange(6,40),"sigma":np.full(34,3/48)}],r_cut=5.0,dim=2):
  if dim == 2:
    tri_cut = int(np.ceil(2.42*r_cut))#rigorous bound
  else:
    print("Dimension not supported in triangulation_radial_func")
    #TODO: even if I don't have a theorem, get an empirical bound
    raise SystemExit

  for i in interesting_particles:
   neighb_list_temp = phom.find_neighbors(i,comp, 2*tri_cut, target_dim=0)
   neighb_list = []
   neighb_dist = []
   for j in neighb_list_temp:
     r = np.linalg.norm(embed.get_diff(embed.get_vpos(i),embed.get_vpos(j)))
     if r < r_cut:
       neighb_list.append(j)
       neighb_dist.append(r)
   neighb_dist = np.array(neighb_dist)
   array_output = all_classes_radial_func(neighb_list,neighb_dist,classes=classes,func=func,params=params)
   sf_names = []
   for c in range(len(params)):
       sf_names += ["g"+chr(ord("A")+c)+"_"+str(i)  for i in range(len(params[c][list(params[c].keys())[0]]))]
   temp_df = pd.DataFrame(array_output.reshape(1,-1),index=[i],columns=sf_names)
   temp_df.insert(0,"particle",i)
   try:
     output_df = output_df.append(temp_df,ignore_index=True)
   except Exception as e:
     output_df = temp_df
  return output_df

def bp_angular_sf(vecs,R=1.0,lamb=1.0, zeta=1.0):
    r1 = np.dot(vecs[0],vecs[0])
    r2 = np.dot(vecs[1], vecs[1])
    c = np.dot(vecs[0],vecs[1])/ np.sqrt(r1*r2)
    R2sum = r1 + r2 + np.dot(vecs[1]-vecs[0], vecs[1]-vecs[0])
    return np.exp(-R2sum/R**2)*(1+lamb*c)**zeta

#invalid defaults
def broadcast_angular_func(neighb_vecs,func=bp_angular_sf,params={"mean":0.5*np.arange(2,11),"sigma":np.full(9,0.5)}):
   output = np.zeros(len(params[list(params.keys())[0]]))#access an arbitrary element of dictionary
   for vecs in neighb_vecs:
     output += func(vecs,**params)
   return output


def all_classes_angular_func(neighb_pair_list,neighb_vecs,classes=np.zeros(4096),func=bp_angular_sf,params=[{"mean":0.25*np.arange(4,22),"sigma":np.full(18,0.25)}, {"mean":0.35*np.arange(4,22),"sigma":np.full(18,0.35)}]):
    N_classes = len(params) #so here, NOT = number of particle classes. 2 particle classes = 3 descriptor classes
    output = np.zeros((N_classes,len(params[0][list(params[0].keys())[0]])))
    for c in range(N_classes):
      output[c] += broadcast_angular_func(neighb_vecs[np.where(np.sum(classes[neighb_pair_list], axis=1)==c)[0]],func=func,params=params[c])

    return output

#again, defaults don't work
def triangulation_angular_func(embed,comp,interesting_particles=np.arange(2048),classes=np.concatenate((np.zeros(2048,dtype=int),np.full(2048,1,dtype=int))),func=gaussian_sf,params=[{"mean":(5/96)*np.arange(6,40),"sigma":np.full(34,5/96)},{"mean":(3/48)*np.arange(6,40),"sigma":np.full(34,3/48)}],r_cut=5.0,dim=2, key_prefix=""):
  if dim == 2:
    tri_cut = int(np.ceil(2.42*r_cut))#rigorous bound
  else:
    print("Dimension not supported in triangulation_angular_func")
    #TODO: even if I don't have a theorem, get an empirical bound
    raise SystemExit

  for i in interesting_particles:
   neighb_list_temp = list(phom.find_neighbors(i,comp, 2*tri_cut, target_dim=0))
   neighb_list = []
   neighb_dist = []
   for j_ind in range(len(neighb_list_temp)):
     j = neighb_list_temp[j_ind]
     r = np.linalg.norm(embed.get_diff(embed.get_vpos(i),embed.get_vpos(j)))
     if r < r_cut:
       for k_ind in range(j_ind+1, len(neighb_list_temp)):
         k = neighb_list_temp[k_ind]
         r2 = np.linalg.norm(embed.get_diff(embed.get_vpos(i),embed.get_vpos(k)))
         if r2 < r_cut:
           neighb_list.append([j,k])
           neighb_dist.append([embed.get_diff(embed.get_vpos(i),embed.get_vpos(j)), embed.get_diff(embed.get_vpos(i),embed.get_vpos(k))])
   neighb_dist = np.array(neighb_dist)
   array_output = all_classes_angular_func(neighb_list,neighb_dist,classes=classes,func=func,params=params)
   sf_names = []
   def pair(i):
       if i == 0:
           out = "AA"
       if i == 1:
           out = "AB"
       if i == 2:
           out = "BB"
       return out
   for c in range(len(params)):
       sf_names += ["ang"+key_prefix+"-"+pair(c)+"_"+str(i)  for i in range(len(params[c]["R"]))]
   temp_df = pd.DataFrame(array_output.reshape(1,-1),index=[i],columns=sf_names)
   temp_df.insert(0,"particle",i)
   try:
     output_df = output_df.append(temp_df,ignore_index=True)
   except Exception as e:
     output_df = temp_df
  return output_df
