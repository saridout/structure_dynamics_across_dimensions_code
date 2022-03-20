import sys, os
sys.path.insert(0, '../')
sys.path.insert(0, '../python_src/')

import numpy as np
import scipy as sp

import scipy.sparse as sparse
import scipy.optimize as opt
import scipy.spatial as spatial

import queue

import phat


def construct_filtration(comp, filtration, label=True):
        
    Q = queue.PriorityQueue()
    
    if label:
        for i in range(comp.ncells):
            Q.put((filtration.get_total_order(comp.get_label(i)), i))
    else:
        for i in range(comp.ncells):
            Q.put((filtration.get_total_order(i), i))
        
    while not Q.empty():
        (order, c) = Q.get()
        yield c

def compute_persistence(comp, filtration, extended=False, birth_cycles=False, optimal_cycles=False,
                       weights=None, relative_cycles=False):
    
    columns = []
    
    cell_to_col = {}
    col_to_cell = []
       
    icol = 0
        
    for ci in filtration:
        
        # Z2 coefficients
        if not optimal_cycles:
            col = set()
            for cj in comp.get_facets(ci):
                if comp.regular:
                    col.add(cell_to_col[cj])
                else:
                    if comp.get_coeffs(ci)[cj] % 2 != 0:
                        col.add(cell_to_col[cj])
                
            if birth_cycles:
                columns.append(col)
            else:
                columns.append((comp.get_dim(ci), sorted(col)))
                
        # general coefficients
        else:
            col = {}
            for cj in comp.get_facets(ci):
                if comp.get_facets(ci)[cj] != 0:
                    col[cell_to_col[cj]] = comp.get_facets(ci)[cj]

            columns.append(col)
        
        cell_to_col[ci] = icol
        icol += 1
        col_to_cell.append(ci)

        
    if not optimal_cycles and not birth_cycles:
    
        boundary_matrix = phat.boundary_matrix(columns=columns)

        alive = set(range(len(columns)))

        pairs = []
        for (i, j) in boundary_matrix.compute_persistence_pairs():
            alive.discard(i)
            alive.discard(j)

            ci = col_to_cell[i]
            cj = col_to_cell[j]

            pairs.append((ci, cj))


        for ci in alive:
            pairs.append((col_to_cell[ci], None))
        
        return pairs
    
    elif not optimal_cycles and birth_cycles:
    
        # pivot row of each column if it has one
        pivot_row = {}
        # row to reduced column with pivot in that row
        pivot_col = {}
        
        g = []
            
        for j in range(len(columns)):
            
            g.append({j})
            
            if len(columns[j]) > 0:
                pivot_row[j] = max(columns[j])
            
            while len(columns[j]) > 0 and pivot_row[j] in pivot_col:
                
                l = pivot_col[pivot_row[j]]
                columns[j] ^= columns[l]
                
                g[j] ^= g[l]
                
                if len(columns[j]) > 0:
                    pivot_row[j] = max(columns[j])
                else:
                    del pivot_row[j]
                
                
                    
            if len(columns[j]) > 0:
                pivot_col[pivot_row[j]] = j
 
        
        alive = set(range(len(columns)))

        pairs = []
        for (j, i) in pivot_row.items():
            alive.discard(i)
            alive.discard(j)

            ci = col_to_cell[i]
            cj = col_to_cell[j]
            
            pairs.append((ci, cj))


        for ci in alive:
            pairs.append((col_to_cell[ci], None))

        bcycles = {}
        for i in range(len(columns)):
            if len(columns[i]) > 0 or comp.get_dim(col_to_cell[i]) == 0:
                continue

            ci = col_to_cell[i]

            bcycles[ci] = set()
            for j in g[i]:
                bcycles[ci].add(col_to_cell[j])
                                            
        return (pairs, bcycles)
    
        
    elif optimal_cycles:
                
        # pivot row of each column if it has one
        pivot_row = {}
        # row to reduced column with pivot in that row
        pivot_col = {}
        
        g = []
        
        ocycles = {}
        
        cell_counts = {i:0 for i in range(1, comp.dim+1)}
        
        x_to_cell = {i:{} for i in range(1, comp.dim+1)}
        cell_to_x = {i:{} for i in range(1, comp.dim+1)}
        
        B = {i+1:{} for i in range(1, comp.dim+1)}
        
        Z = {i:{} for i in range(1, comp.dim+1)}
            
        for j in range(len(columns)):
            
            g.append({j:1})
            
            if len(columns[j]) > 0:
                pivot_row[j] = max(columns[j])
               
            d = comp.get_dim([col_to_cell[j]])
            if d > 0:
                x_to_cell[d][cell_counts[d]] = col_to_cell[j]
                cell_to_x[d][col_to_cell[j]] = cell_counts[d]
                cell_counts[d] += 1
                
            
            while len(columns[j]) > 0 and pivot_row[j] in pivot_col:
                
                p = pivot_row[j]
                l = pivot_col[p]
                
                r = 1.0 * columns[j][p] / columns[l][p]
                
                for k in columns[l]:
                    columns[j][k] = columns[j].get(k, 0) - r * columns[l][k]
                    if columns[j][k] == 0.0:
                        del columns[j][k]
                    
                for k in g[l]:
                    g[j][k] = g[j].get(k, 0) - r * g[l][k]
                    if g[j][k] == 0.0:
                        del g[j][k]
                                    
                if len(columns[j]) > 0:
                    pivot_row[j] = max(columns[j])
                else:
                    del pivot_row[j]
                
            if len(columns[j]) == 0 and d > 0:
                
                c = np.ones(2*(cell_counts[d]+len(B[d+1])+len(Z[d])), float)
                if weights is not None:
                    for k in range(cell_counts[d]):
                        c[k] = weights[x_to_cell[d][k]]
                        c[k+cell_counts[d]] = weights[x_to_cell[d][k]]
                else:
                    c[0:2*cell_counts[d]] = 1.0
                
                A_i = []
                A_j = []
                A_val = []
                
                for k in range(cell_counts[d]):
                    A_i.append(k)
                    A_j.append(k)
                    A_val.append(1.0)
                    
                    A_i.append(k)
                    A_j.append(k+cell_counts[d])
                    A_val.append(-1.0)
                    
                                    
                for bi, kj in enumerate(B[d+1]):
                    for ki in B[d+1][kj]:
                        A_i.append(cell_to_x[d][ki])
                        A_j.append(2*cell_counts[d] + bi)
                        A_val.append(-B[d+1][kj][ki])
                        
                        A_i.append(cell_to_x[d][ki])
                        A_j.append(2*cell_counts[d] + len(B[d+1]) + bi)
                        A_val.append(B[d+1][kj][ki])
                 
                for zi, kj in enumerate(Z[d]):
                    for ki in Z[d][kj]:
                        A_i.append(cell_to_x[d][ki])
                        A_j.append(2*cell_counts[d] + 2*len(B[d+1]) + zi)
                        A_val.append(Z[d][kj][ki])

                        A_i.append(cell_to_x[d][ki])
                        A_j.append(2*cell_counts[d] + 2*len(B[d+1]) + len(Z[d]) + zi)
                        A_val.append(-Z[d][kj][ki])

                
                b_eq = np.zeros(cell_counts[d], float)
                # print(g[j])
                for k in g[j]:
                    b_eq[cell_to_x[d][col_to_cell[k]]] = g[j][k]
                
                res = opt.linprog(c, A_eq=sparse.coo_matrix((A_val, (A_i, A_j))), 
                                  b_eq=b_eq, method='interior-point', 
                                  options={'disp':False, 'maxiter': 100000, 'sparse': True, 
                                           'ip': False, 'permc_spec':'COLAMD'})
                
                if res.status != 0:
                    print(res)
                                
                
                print(j, "/", len(columns), "size", len(c), "nit", res.nit, "Change", np.sum([weights[x_to_cell[d][k]] for k in np.where(b_eq != 0)[0]]), "->", int(round(res.fun)))
                
                z = res.x[0:cell_counts[d]] - res.x[cell_counts[d]:2*cell_counts[d]]
                
                # print(z)
                
                col = {}
                for k in range(cell_counts[d]):
                    h = int(round(z[k]))
                    if h != 0:
                        col[x_to_cell[d][k]] = h
                        
                if relative_cycles:
                    Z[d][j] = col
                    
                ocycles[col_to_cell[j]] = set(col.keys())
                    
            elif len(columns[j]) > 0:
                p = pivot_row[j]
                pivot_col[p] = j
                
                if d > 1:
                    col = {}
                    for k in columns[j]:
                        if columns[j][k] != 0.0:
                            col[col_to_cell[k]] = columns[j][k]
                    B[d][j] = col
                
                    if relative_cycles:
                        del Z[d-1][p]
                            
           
        alive = set(range(len(columns)))

        pairs = []
        for (j, i) in pivot_row.items():
            alive.discard(i)
            alive.discard(j)

            ci = col_to_cell[i]
            cj = col_to_cell[j]

            pairs.append((ci, cj))


        for ci in alive:
            pairs.append((col_to_cell[ci], None))

        if birth_cycles:
            
            bcycles = {}
            for i in range(len(columns)):
                if len(columns[i]) > 0 or comp.get_dim(col_to_cell[i]) == 0:
                    continue

                ci = col_to_cell[i]

                bcycles[ci] = set()
                for j in g[i]:
                    bcycles[ci].add(col_to_cell[j])
                
            return (pairs, bcycles, ocycles)
        else:
            return (pairs, ocycles)
        
        
        