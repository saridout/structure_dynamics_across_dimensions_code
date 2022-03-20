import sys, os
sys.path.insert(0, '../../lib_persistent_homology/')
sys.path.insert(0, '../../lib_persistent_homology/python_src/')

import numpy as np
import numpy.linalg as la
import scipy as sp
import pandas as pd

import phom

def get_configuration(state, index):
    
    DIM = len(state.dimensions['dim'])

    NP = len(state.dimensions['NP'])

    pos = state.variables['pos'][index]

    rad2 = state.variables['rad'][index]**2

    box_mat = np.array(state.variables['BoxMatrix'][index].reshape((DIM, DIM))).T

    if DIM == 2:
        embed = phom.Embedding2D(NP, pos, box_mat, True)
    elif DIM == 3:
        embed = phom.Embedding3D(NP, pos, box_mat, True)
    elif DIM == 4:
        embed = phom.Embedding4D(NP, pos, box_mat, True)
    elif DIM == 5:
        embed = phom.Embedding5D(NP, pos, box_mat, True)    
    return (embed, rad2)

def get_configuration_misaki(filename):
    DIM = 2
    stuff = np.loadtxt(filename)
    NP = int(stuff[0,0])
    strain = stuff[0,2]
    L = stuff[0,1]
    rad2 = stuff[1:,2]**2
    pos = stuff[1:,:2]
    for i in range(NP):
      if pos[i,0] < strain*pos[i,1]:
        pos[i,0] += L

    box_mat = np.array([[L,L*strain],[0,L]])
    inv = np.linalg.inv(box_mat)
    for i in range(NP):
      pos[i] = np.dot(inv,pos[i])

    print(np.min(pos),np.max(pos))
    embed = phom.Embedding2D(NP, pos.flatten(), np.asfortranarray(box_mat), True)
    return (embed, rad2)

def get_configuration_sylvain(filename, radfilename):
    DIM = 2
    f = open(filename, "r")
    for i in range(3):
        blah = f.readline()
    NP = int(f.readline())
    f.readline()
    stuff = np.loadtxt(f, max_rows = 3)
    strain = stuff[0,2] / stuff[1,1]
    L = stuff[1,1]
    box_mat = np.array([[L,L*strain],[0,L]])
    f.readline()
    stuff = np.loadtxt(f)
    pos = np.zeros((len(stuff),2))
    for i in range(NP):
       pos[int(stuff[i,0])-1] = stuff[i,1:]

    box_mat = np.array([[L,L*strain],[0,L]])
    inv = np.linalg.inv(box_mat)
    for i in range(NP):
      pos[i] = np.dot(inv,pos[i])

    print(np.min(pos),np.max(pos))
    #now get radii
    f.close()
    stuff = np.loadtxt(radfilename,dtype=int)
    r1 = 0.3090169943749474241022934171828190588601545899028814310677243114
    r2 = 0.5877852522924731291687059546390727685976524376431459910722724808
    rad2 = np.zeros(NP)
    for i in range(NP):
       rad2[stuff[i,0]-1]= r1**2 if stuff[i,1]==1 else r2**2
    embed = phom.Embedding2D(NP, pos.flatten(), np.asfortranarray(box_mat), True)
    return (embed, rad2)


def get_neighborhood(particle, embed, rad2, max_neigh_dist):
            
    if embed.dim == 2:
        (ieighborhood, neigh_pos) = phom.get_point_neighborhood_2D(particle, max_neigh_dist, embed)
    elif embed.dim == 3:
        (neighborhood, neigh_pos) = phom.get_point_neighborhood_3D(particle, max_neigh_dist, embed)
        
    neigh_pos = np.array(neigh_pos).flatten()  
    
    center = embed.get_vpos(particle)
    
    for i in range(len(neighborhood)):
        neigh_pos[embed.dim*i: embed.dim*(i+1)] -= center - 0.5*np.ones(embed.dim)
    
    
    
    if embed.dim == 2:
        neigh_embed = phom.Embedding2D(len(neighborhood), neigh_pos, np.array(embed.box_mat), False)
    elif embed.dim == 3:
        neigh_embed = phom.Embedding3D(len(neighborhood), neigh_pos, np.array(embed.box_mat), False)
    
    return (np.array(neighborhood), neigh_embed, rad2[neighborhood])

    
def construct_triangulation(embed, rad2):
                
    if embed.dim == 2:
        comp = phom.construct_alpha_complex_2D(embed, rad2, dim_cap=1)
    elif embed.dim == 3:
        comp = phom.construct_alpha_complex_3D(embed, rad2, dim_cap=1)
    elif embed.dim == 4:
        comp = phom.construct_alpha_complex_4D(embed, rad2, dim_cap=1)
    elif embed.dim == 5:
        comp = phom.construct_alpha_complex_5D(embed, rad2, dim_cap=1)
    return comp
        
        
def calc_D2min(comp, mode, embed, layers=1):
                
        
    if embed.dim == 2: 
        D2min, strain = phom.calc_delaunay_D2min_strain_2D(mode, comp, embed, max_dist=2*layers)
    elif embed.dim == 3:
        D2min, strain = phom.calc_delaunay_D2min_strain_3D(mode, comp, embed, max_dist=2*layers)
    elif embed.dim == 4:
        D2min, strain = phom.calc_delaunay_D2min_strain_4D(mode, comp, embed, max_dist=2*layers)
    elif embed.dim == 5:
        D2min, strain = phom.calc_delaunay_D2min_strain_5D(mode, comp, embed, max_dist=2*layers)
    return D2min

def calc_D2min_straintensor(comp, mode, embed, layers=1):

    if embed.dim == 2:
        stuff = phom.calc_voronoi_D2min_straintensor_2D(mode, comp, embed, max_dist=2*layers)
    elif embed.dim == 3:
        stuff = phom.calc_voronoi_D2min_straintensor_3D(mode, comp, embed, max_dist=2*layers)
    elif embed.dim == 4:
        stuff = phom.calc_voronoi_D2min_straintensor_4D(mode, comp, embed, max_dist=2*layers)
    elif embed.dim == 5:
        stuff = phom.calc_voronoi_D2min_straintensor_5D(mode, comp, embed, max_dist=2*layers)
    return stuff
    
def calc_D2min_contacts(comp, mode, embed, rad2):
    
    r2norm = np.min(rad2)
    
    if embed.dim == 2:
        alpha_vals = phom.calc_alpha_vals_2D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 3:
        alpha_vals = phom.calc_alpha_vals_3D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 4:
        alpha_vals = phom.calc_alpha_vals_4D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 5:
        alpha_vals = phom.calc_alpha_vals_5D(comp, embed, rad2, alpha0=-r2norm)
        
    is_contact = np.where(alpha_vals[comp.dcell_range[1][0]:comp.dcell_range[1][1]] <= 0.0, True, False)
    
    if embed.dim == 2:
        D2min = phom.calc_voronoi_D2min_2D(mode, comp, embed, is_contact)
    elif embed.dim == 3:
        D2min = phom.calc_voronoi_D2min_3D(mode, comp, embed, is_contact)
    elif embed.dim == 4:
        D2min = phom.calc_voronoi_D2min_4D(mode, comp, embed, is_contact)
    elif embed.dim == 5:
        D2min = phom.calc_voronoi_D2min_5D(mode, comp, embed, is_contact)

    return D2min
    
    
def normalize_D2min(D2min, comp, embed, layers=2):
    
    D2min_normed = np.zeros_like(D2min)
    for vi in range(*comp.dcell_range[0]):

        verts = list(phom.find_neighbors(vi, comp, 2*layers, target_dim=0))

        D2min_normed[vi] = D2min[vi] / np.mean(D2min[verts])
        
    return D2min_normed
    
        
def find_local_extrema(height, comp, layers=1):
    
    return phom.find_local_extrema(height, comp, max_dist=2*layers)


def remove_rattlers(comp, embed, rad2):
    
    rattlers = []
    neigh_pos = []
    new_rad2 = []
    
    for p in range(embed.NV):
        edges = comp.get_cofacets(p)
        
        Z = 0
        
        for e in edges:
            
            verts = list(comp.get_facets(e))
            
            vi = verts[0]
            vj = verts[1]
            
            posi = embed.get_vpos(vi)
            posj = embed.get_vpos(vj)
            
            bvec = embed.get_diff(posi, posj)
            
            if la.norm(bvec) < np.sqrt(rad2[vi]) + np.sqrt(rad2[vj]):
                Z += 1
                
        if Z < embed.dim + 1:
            rattlers.append(p)
        else:
            neigh_pos.extend(embed.get_vpos(p))
            new_rad2.append(rad2[p])
        
    if embed.dim == 2:
        new_embed = phom.Embedding2D(embed.NV - len(rattlers), np.array(neigh_pos), np.array(embed.box_mat), embed.periodic)
    elif embed.dim == 3:
        new_embed = phom.Embedding3D(embed.NV - len(rattlers), np.array(neigh_pos), np.array(embed.box_mat), embed.periodic)
    elif embed.dim == 4:
        new_embed = phom.Embedding4D(embed.NV - len(rattlers), np.array(neigh_pos), np.array(embed.box_mat), embed.periodic)
    elif embed.dim == 5:
        new_embed = phom.Embedding5D(embed.NV - len(rattlers), np.array(neigh_pos), np.array(embed.box_mat), embed.periodic)

    new_comp = construct_triangulation(new_embed, new_rad2)
    
    return (rattlers, new_comp, new_embed, np.array(new_rad2))
    

def find_rattlers(particles, comp, embed, rad2):
    
    rattlers = []
    
    for p in particles:
        edges = comp.get_cofacets(p)
        
        Z = 0
        
        for e in edges:
            
            verts = list(comp.get_facets(e))
            
            vi = verts[0]
            vj = verts[1]
            
            posi = embed.get_vpos(vi)
            posj = embed.get_vpos(vj)
            
            bvec = embed.get_diff(posi, posj)
            
            if la.norm(bvec) < np.sqrt(rad2[vi]) + np.sqrt(rad2[vj]):
                Z += 1
                
        if Z < embed.dim + 1:
            rattlers.append(p)
        
        
    return rattlers
            
            
    
        
        

def get_network(embed, comp):
    
    NE = comp.ndcells[1]
    edgei = []
    edgej = []
    
    for c in range(*comp.dcell_range[1]):
        
        facets = comp.get_facets(c)
        edgei.append(facets[0])
        edgej.append(facets[1])
            
    DIM = embed.dim
    node_pos = np.zeros(DIM*embed.NV, float)
    for i in range(embed.NV):
        node_pos[DIM*i:DIM*i+DIM] = embed.get_pos(i)
            
    return (embed.NV, node_pos, NE, edgei, edgej, embed.box_mat.diagonal())

    
#for a numpy array X defined on the particles, compute for each edge e_ij Y := X_i - X_j
def compute_edge_differences(X, embed,comp):
    DIM = embed.dim
    NP = comp.ndcells[0]    
    NE = comp.ndcells[1]
    
    Y = np.zeros(NE)
    
    for c in range(comp.dcell_begin[1], comp.dcell_begin[1]+comp.ndcells[1]):
        
        facets = comp.get_facets(c)
        i = np.min(facets)
        j = np.max(facets)
        Y[c-comp.dcell_begin[1]] = X[j] - X[i]
        
    return Y    
        
    
    
    
