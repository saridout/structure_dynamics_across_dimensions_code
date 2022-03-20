import numpy as np

import phom

#voro: phom.Voronoi2D object

#vector from particle centers to centroids, flattened (cx1, cy1, cx2, cy2, ....)
def c_vec(voro):
    centroids = voro.get_cell_centroids()
    centers = np.zeros(len(centroids))
    for i in range(len(centers)//2):
       centers[2*i:(2*i+2)]  = voro.embed.get_pos(i)

    return centroids - centers


#given a vector field defined on the particles and the indices of a triangle, compute discrete divergence
def compute_triangle_divergence(field, indices, voro):
    full_indices = np.concatenate([[2*i, 2*i+1] for i in indices])
    RHS = field[full_indices]


    mat = np.zeros((6,6))

    for i in range(3):
        mat[2*i, 2:4] = voro.embed.get_pos(indices[i])
        mat[2*i+1, 4:] = voro.embed.get_pos(indices[i])
        mat[2*i, 0] = 1
        mat[2*i+1, 1] = 1
    d = np.linalg.solve(mat, RHS)

    return d[2] + d[5]



#compute Qk for each triangle
def compute_triangle_Qk(voro):
   N_tri = voro.comp.ndcells[2]
   Qk = np.zeros(N_tri)
   areas = voro.get_cell_areas()
   avg = np.mean(area) 
   C = c_vec(voro)
   for i in range(N_tri):
       particles = np.get_faces(voro.comp.dcell_range[0,0]+i,0) 
       Qk[i] = (areas[i]/avg)*compute_triangle_divergence(C, particles, voro)
