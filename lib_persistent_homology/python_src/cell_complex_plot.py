import numpy as np
import scipy as sp
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import collections as collections
import matplotlib.patches as patches
import scipy.spatial as spatial


import sys
sys.path.insert(0, '../../lib_persistent_homology/')

import phom


gray = "#969696"


def show_network(ax, comp, embed, styles={}, alpha=1.0, boundary_cutoff=0.1, zorder=None, kwargs=dict()):
    
    box_mat = embed.box_mat
    L = np.diagonal(box_mat)
        
    image_offsets = [np.array([0.0, 0.0]),
                    np.array([1.0, 0.0]),
                    np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, -1.0]),
                    np.array([1.0, 1.0]),
                    np.array([-1.0, 1.0]),
                    np.array([1.0, -1.0]),
                    np.array([-1.0, -1.0])]
    
    edges = []
    edge_index = []
    for c in range(*comp.dcell_range[1]):
        
        (vi, vj) = comp.get_facets(c)
        
        vposi = embed.get_vpos(vi)
        vposj = embed.get_vpos(vj)
        
        vbvec = embed.get_vdiff(vposi, vposj)
       
            
        posi = box_mat.dot(vposi) / L
        bvec = box_mat.dot(vbvec) / L
        posj = posi + bvec
        
        if embed.periodic:
                        
            test_duplicates = ((posi < 0.0).any() or (posi > 1.0).any()
                              or (posj < 0.0).any() or (posj > 1.0).any())
            
            
            if test_duplicates:
                for offset in image_offsets:
                    oposi = box_mat.dot(vposi+offset) / L
                    oposj = oposi+bvec
                    
                    if ((oposi > -boundary_cutoff).all() and (oposi < 1.0+boundary_cutoff).all()
                        and (oposj > -boundary_cutoff).all() and (oposj < 1.0+boundary_cutoff).all()):
                        edges.append([tuple(oposi),tuple(oposj)])
                        edge_index.append(comp.get_label(c))
                    
            else:
                edges.append([tuple(posi),tuple(posj)])
                edge_index.append(comp.get_label(c))
                
        else:
            edges.append([tuple(posi),tuple(posj)])
            edge_index.append(comp.get_label(c))
                
        
        
        
    ls = []
    colors = []
    lw = []
    
    for i, b in enumerate(edge_index):
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append(gray)
            
            
        if b in styles and 'ls' in styles[b]:
            ls.append(styles[b]['ls'])
        else:
            ls.append('solid')
            
        if b in styles and 'lw' in styles[b]:
            lw.append(styles[b]['lw'])
        else:
            lw.append(2.0)
            
            
            
    lc = collections.LineCollection(edges, linestyle=ls, lw=lw, alpha=alpha, color=colors, zorder=zorder, **kwargs)
    ax.add_collection(lc)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
def show_discs(ax, comp, embed, rad, subset=None, styles={}, alpha=1.0, boundary_cutoff=0.01, zorder=None, edgecolor='k', kwargs=dict()):
    
    box_mat = embed.box_mat
    L = np.diagonal(box_mat)
    
    image_offsets = [np.array([0.0, 0.0]),
                    np.array([1.0, 0.0]),
                    np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, -1.0]),
                    np.array([1.0, 1.0]),
                    np.array([-1.0, 1.0]),
                    np.array([1.0, -1.0]),
                    np.array([-1.0, -1.0])]    
    
    discs = []
    disc_index = []
    
    if subset is not None:
        cell_list = subset
    else:
        cell_list = range(*comp.dcell_range[0])
    
    for c in cell_list:
        
        vi = comp.get_label(c)
        
        vposi = embed.get_vpos(vi)       

        posi = box_mat.dot(vposi) / L
                    
        r = rad[vi] / L[0]
        
        
        if embed.periodic:
            
            test_duplicates = ((posi < r+boundary_cutoff).any() or (posi > 1.0-r-boundary_cutoff).any())
            
            if test_duplicates:
                for offset in image_offsets:
                    oposi = box_mat.dot(vposi+offset) / L
                    
                    if ((oposi > -r-boundary_cutoff).all() and (oposi < 1.0+r+boundary_cutoff).all()):
                        discs.append(patches.Circle(oposi, r))
                        disc_index.append(vi)
                    
            else:
                discs.append(patches.Circle(posi, r))
                disc_index.append(vi)
                
        else:
            discs.append(patches.Circle(posi, r))
            disc_index.append(vi)
        
        
        
        
    colors = []
    
    for i, b in enumerate(disc_index):
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append('white')
            
            
            
    pc = collections.PatchCollection(discs, edgecolor=edgecolor, linewidth=0.5, alpha=alpha, facecolors=colors, zorder=zorder, **kwargs)
    ax.add_collection(pc)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    
    
def show_verts(ax, comp, embed, nodes, styles={}, alpha=1.0, zorder=None, marker='o', shadow=False, kwargs=dict()):
    
    box_mat = embed.box_mat
    L = np.diagonal(box_mat)
    
    image_offsets = [np.array([0.0, 0.0]),
                    np.array([1.0, 0.0]),
                    np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, -1.0]),
                    np.array([1.0, 1.0]),
                    np.array([-1.0, 1.0]),
                    np.array([1.0, -1.0]),
                    np.array([-1.0, -1.0])]  
    
    
    x = []
    y = []
    vert_index = []
    
    for c in nodes:
        
        vi = comp.get_label(c)
        
        vposi = embed.get_vpos(vi)       

        posi = box_mat.dot(vposi) / L
                            
        
        if embed.periodic:
            
            test_duplicates = ((posi < 0.0).any() or (posi > 1.0).any())
            
            if test_duplicates:
                for offset in image_offsets:
                    oposi = box_mat.dot(vposi+offset) / L
                    
                    if ((oposi > 0.0).all() and (oposi < 1.0).all()):
                        x.append(oposi[0])
                        y.append(oposi[1])
                        vert_index.append(vi)
                    
            else:
                x.append(posi[0])
                y.append(posi[1])
                vert_index.append(vi)
                
        else:
            x.append(posi[0])
            y.append(posi[1])
            vert_index.append(vi)
        
        
        
        
    colors = []
    sizes = []
    
    for i, b in enumerate(vert_index):
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append('k')
            
        if b in styles and 'size' in styles[b]:
            sizes.append(styles[b]['size'])
        else:
            sizes.append(200)
            
            
    if shadow:
        ax.scatter(np.array(x), np.array(y), marker=marker , s=1.25*np.array(sizes), facecolor='#636363', alpha=0.5, zorder=zorder)
    
    ax.scatter(x, y, marker=marker , s=sizes, facecolor=colors, alpha=1.0, linewidths=0.0, zorder=zorder)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
  
    
def show_vec_field(ax, comp, embed, vec_field, subset=None, zorder=None, boundary_cutoff=0.01, color='k', kwargs=dict()):
    
    box_mat = embed.box_mat
    L = np.diagonal(embed.box_mat)
    DIM = embed.dim
    
    image_offsets = [np.array([0.0, 0.0]),
                    np.array([1.0, 0.0]),
                    np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, -1.0]),
                    np.array([1.0, 1.0]),
                    np.array([-1.0, 1.0]),
                    np.array([1.0, -1.0]),
                    np.array([-1.0, -1.0])]    
    
    X = []
    Y = []
    U = []
    V = []
    norm = []
    
    
    if subset is not None:
        cell_list = subset
    else:
        cell_list = range(*comp.dcell_range[0])
    
    for c in cell_list:
 
        vi = comp.get_label(c)
        
        vpos = embed.get_vpos(vi)
        
        pos = embed.get_pos(vi) / L
        
        u = vec_field[DIM*vi:DIM*vi+DIM] / L
        norm.append(la.norm(u))
        
        
        if embed.periodic:
            test_duplicates = ((pos < boundary_cutoff).any() or (pos > 1.0-boundary_cutoff).any())
            
            if test_duplicates:
                for offset in image_offsets:
                    opos = box_mat.dot(vpos+offset) / L
                    
                    # this logic may be off
                    if ((opos > -boundary_cutoff).all() and (opos < 1.0+boundary_cutoff).all()):
                        X.append(opos[0])
                        Y.append(opos[1])
                        U.append(u[0])
                        V.append(u[1])
                        
            else:
                X.append(pos[0])
                Y.append(pos[1])
                U.append(u[0])
                V.append(u[1])
                
        else:
            X.append(pos[0])
            Y.append(pos[1])
            U.append(u[0])
            V.append(u[1])
        
        
    asort = np.argsort(norm)
    X = np.array(X)[asort]
    Y = np.array(Y)[asort]
    U = np.array(U)[asort]
    V = np.array(V)[asort]
        
        
    ax.quiver(X, Y, U, V, units='xy', scale=1.0, zorder=None, color=color, **kwargs)


def show_patches(ax, comp, embed, subset=None, styles={}, alpha=1.0, boundary_cutoff=0.01, zorder=None, kwargs=dict()):
    
    box_mat = embed.box_mat
    L = np.diagonal(box_mat)
    
    image_offsets = [np.array([0.0, 0.0]),
                    np.array([1.0, 0.0]),
                    np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, -1.0]),
                    np.array([1.0, 1.0]),
                    np.array([-1.0, 1.0]),
                    np.array([1.0, -1.0]),
                    np.array([-1.0, -1.0])]    
    
    patch_list = []
    patch_index = []
    
    if subset is not None:
        cell_list = subset
    else:
        cell_list = range(*comp.dcell_range[2])
    
    for c in cell_list:
        
        verts = list(comp.get_faces(c, 0))
        
        vposi = embed.get_vpos(verts[0])
        posi = box_mat.dot(vposi) / L
        
        corners = np.zeros([len(verts), 2], float)
        corners[0] = posi
        
        for j in range(1, len(verts)):
            vposj = embed.get_vpos(verts[j])
            vbvec = embed.get_vdiff(vposi, vposj)
            
            bvec = box_mat.dot(vbvec) / L
            posj = posi + bvec
            corners[j] = posj
            
        if embed.periodic:
            test_duplicates = ((posi < boundary_cutoff).any() or (posi > 1.0-boundary_cutoff).any())
            
            if test_duplicates:
                for offset in image_offsets:
                    oposi = box_mat.dot(vposi+offset) / L
                    
                    
                    # this logic may be off
                    if ((oposi > -boundary_cutoff).all() and (oposi < 1.0+boundary_cutoff).all()):
                        
                        corners_tmp = np.copy(corners)
                        # first coordinate may be wrong in corners_tmp since first coordinate does not get offset
                        for j in range(1, len(verts)):
                            vposj = embed.get_vpos(verts[j])
                            vbvec = embed.get_vdiff(vposi+offset, vposj)
                            bvec = box_mat.dot(vbvec) / L
                            posj = oposi + bvec
                            corners_tmp[j] = posj
                        
                        
                        patch_list.append(patches.Polygon(corners_tmp))
                        patch_index.append(comp.get_label(c))
                    
            else:
                patch_list.append(patches.Polygon(corners))
                patch_index.append(comp.get_label(c))
                
        else:
            patch_list.append(patches.Polygon(corners))
            patch_index.append(comp.get_label(c))
        
        
        
    colors = []    
    for i, b in enumerate(patch_index):
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append(gray)
            
    pc = collections.PatchCollection(patch_list, color=colors, zorder=zorder, alpha=alpha, **kwargs)
    ax.add_collection(pc)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    

def show_voronoi(ax, comp, embed, subset=None, styles={}, alpha=1.0, boundary_cutoff=0.01, zorder=None, kwargs=dict()):
    
    
    box_mat = embed.box_mat
    L = np.diagonal(box_mat)
    
    
    image_offsets = [np.array([0.0, 0.0]),
                    np.array([1.0, 0.0]),
                    np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, -1.0]),
                    np.array([1.0, 1.0]),
                    np.array([-1.0, 1.0]),
                    np.array([1.0, -1.0]),
                    np.array([-1.0, -1.0])]    
    
    
    vert_pos = np.zeros([9*embed.NV, 2], float)
    for vi in range(embed.NV):
        for i, image in enumerate(image_offsets):
            vert_pos[embed.NV*i+vi, :] = box_mat.dot(embed.get_vpos(vi)+image) / L
            
            
    vor = spatial.Voronoi(vert_pos)
        
    patch_list = []
    patch_index = []
    
    if subset is not None:
        cell_list = subset
    else:
        cell_list = range(*comp.dcell_range[0])
    
    for vi in cell_list:
        region = vor.regions[vor.point_region[vi]]
        
        corners = np.zeros([len(region), 2], float)
        
        for j in range(len(region)):
            corners[j] = vor.vertices[region[j]]
            
          
            
        if embed.periodic:
            
            vposi = embed.get_vpos(vi)
            posi = box_mat.dot(vposi) / L
            
            test_duplicates = ((posi < boundary_cutoff).any() or (posi > 1.0-boundary_cutoff).any())
            
            if test_duplicates:
                                
                for offset in image_offsets:
                    oposi = box_mat.dot(vposi+offset) / L
                    
                    
                    # this logic may be off
                    if ((oposi > -boundary_cutoff).all() and (oposi < 1.0+boundary_cutoff).all()):
                        
                        corners_tmp = np.copy(corners)
                        # first coordinate may be wrong in corners_tmp since first coordinate does not get offset
                        for j in range(len(region)):
                            
                            vposj = vor.vertices[region[j]]
                            vbvec = embed.get_vdiff(vposi+offset, vposj)
                            bvec = box_mat.dot(vbvec) / L
                            posj = oposi + bvec
                            
                            corners_tmp[j] = posj
                            
                        
                        patch_list.append(patches.Polygon(corners_tmp))
                        patch_index.append(comp.get_label(vi))
                    
            else:
                                
                patch_list.append(patches.Polygon(corners))
                patch_index.append(comp.get_label(vi))
                
        else:
            patch_list.append(patches.Polygon(corners))
            patch_index.append(comp.get_label(vi))
                
                    
        
    colors = []    
    for i, b in enumerate(patch_index):
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append(gray)
            
    pc = collections.PatchCollection(patch_list, color=colors, zorder=zorder, alpha=alpha, **kwargs)
    ax.add_collection(pc)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    
