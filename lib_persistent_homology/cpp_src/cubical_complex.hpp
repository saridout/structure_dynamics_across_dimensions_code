#ifndef CUBICALCOMPLEX_HPP
#define CUBICALCOMPLEX_HPP
 
#include "eigen_macros.hpp"
#include "cell_complex.hpp"
#include "filtration.hpp"

#include <map>
#include <algorithm>
#include <vector>   
#include "math.h"
    


CellComplex construct_cubical_complex(std::vector<int> &shape, bool oriented, bool dual) {
    
    int dim = shape.size();
    
    CellComplex comp(dim, true, false);
    
    // In d dimensions there are (d choose k) cells of dimension k at each vertex. 
    // Each dimension k cell as 2^k vertices.
    
    
    int size = 1;
    std::vector<int> lengths(dim);
    for(int d = 0; d < dim; d++) {
        lengths[d] = dual ? shape[d] + 1: shape[d];
        size *= lengths[d];
        
    }
    
    auto coords_to_index = [&lengths, dim](std::vector<int> multi_index) {
        
        int index = multi_index[0];
        for(int d = 1; d < dim; d++) {
            index *= lengths[d];
            index += multi_index[d];
        }
        
        return index;
        
    };
    
    int ncells = 0;
        
    // Map of cell dimension to map of cell vertex list to cell index
    std::vector<std::map<std::vector<int>, int> > verts_to_cell(dim+2, std::map<std::vector<int>, int>());
    
    std::vector<int> multi_index(dim, 0);
    for(int k = 0; k <= dim; k++) {
        
//         py::print(k, py::arg("flush")=true);
        
        for(int i = 0; i < size; i++) {

            // For dimension k cells, iterate through every combination of k coordinates
            std::vector<bool> mask(k, true);
            mask.resize(dim, false);
            do { 
                
                // Find coords to consider
                bool skip = false;
                std::vector<int> coords;
                for(int m = 0; m < dim; m++) {
                    if(mask[m]) {
                        
                        if(multi_index[m] == lengths[m] - 1) {
                            skip = true;
                            break;
                        }
                        
                        coords.push_back(m);
                    }
                }
                                
                if(skip) {
                    continue;
                }

                // List of vertices comprising cell
                std::vector<int> verts;
                          
                // Go through every subset of coordinates and add vertex
                for(int p = 0; p < (int) pow(2, k); p++) {
                    
                    std::vector<int> tmp_index = multi_index;
                    for(int q = 0; q < k; q++) {
                        if ((p & (1 << q)) != 0) {
                            tmp_index[coords[q]]++;
                        }
                    }
                    
                    verts.push_back(coords_to_index(tmp_index));
                    
                }
                                
                
                // Find facets
                std::vector<int> facets;
                
                std::vector<bool> facet_mask((int) pow(2, k-1), true);
                facet_mask.resize(verts.size(), false);
                do { 
                    
                    // List of vertices comprising facet
                    std::vector<int> facet_verts;

                    // Go through every subset of coordinates and add vertex
                    for(std::size_t p = 0; p < verts.size(); p++) {
                        if(facet_mask[p]) {
                            facet_verts.push_back(verts[p]);
                        }
                    }
                    
                                    
                    if(verts_to_cell[k].count(facet_verts)) {
                        facets.push_back(verts_to_cell[k][facet_verts]);
                    }
                    
                } while(std::prev_permutation(facet_mask.begin(), facet_mask.end()));
                
                            
                
                int label = verts_to_cell[k+1].size();
                
                std::vector<int> coeffs;
                
//                 py::print("facet", label, py::arg("flush")=true);
                
                
                comp.add_cell(label, k, facets, coeffs);
                
                // Store list of cells
                verts_to_cell[k+1][verts] = ncells;
                ncells++;
                
            
            } while(std::prev_permutation(mask.begin(), mask.end()));
            
            
            
            for(int d = dim-1; d >= 0; d--) {
                if(multi_index[d] == lengths[d] - 1) {
                    multi_index[d] = 0;
                } else {
                    multi_index[d]++;
                    break;
                }
                
            }
            
            
        }
        
        
        verts_to_cell[k].clear();
        
    }
    
//     py::print("cofacets", py::arg("flush")=true);
    
    comp.construct_cofacets();
    
    
//     py::print("compressing", py::arg("flush")=true);
    
    comp.make_compressed(); 
        
    return comp;
    
}


CellComplex construct_masked_cubical_complex(std::vector<bool> &mask, std::vector<int> &shape, bool oriented, bool dual) {
      
    CellComplex complete_comp = construct_cubical_complex(shape, oriented, dual);
    
    std::vector<bool> include(complete_comp.ncells, false);
    
    for(int i = 0; i < complete_comp.ncells; i++) {
        
        if((dual && complete_comp.get_dim(i) == complete_comp.dim) 
          || (!dual && complete_comp.get_dim(i) == 0)) {
            
            include[i] = !mask[complete_comp.get_label(i)];

            
        }
    }
    
    if(dual) {
        for(int d = complete_comp.dim-1; d >= 0; d--) {
            for(int i = 0; i < complete_comp.ncells; i++) {
                if(complete_comp.get_dim(i) == d) {
                                        
                    auto range = complete_comp.get_cofacet_range(i);
                    for(auto it = range.first; it != range.second; it++) {
                        
                        if(include[*it]) {
                            include[i] = true;
                        }
                        
                    }
                }
            }
        }
    } else {
        for(int d = 1; d <= complete_comp.dim; d++) {
            for(int i = 0; i < complete_comp.ncells; i++) {
                if(complete_comp.get_dim(i) == d) {
                                        
                    auto range = complete_comp.get_facet_range(i);
                    for(auto it = range.first; it != range.second; it++) {
                        
                        if(include[*it]) {
                            mask[i] = true;
                        }
                        
                    }
                }
            }
        }
    }
      
    
    std::vector<int> complete_to_mask(complete_comp.ncells);
    int index = 0;
    for(int i = 0; i < complete_comp.ncells; i++) {
        if(include[i]) {
            complete_to_mask[i] = index;
            index++;
        }
    }
    
    CellComplex mask_comp(complete_comp.dim, true, oriented);
    
    for(int i = 0; i < complete_comp.ncells; i++) {
        if(include[i]) {
            
            std::vector<int> facets;
            auto facet_range = complete_comp.get_facet_range(i);
            for(auto it = facet_range.first; it != facet_range.second; it++) {
                facets.push_back(complete_to_mask[*it]);
            }
            std::vector<int> coeffs;
            auto coeff_range = complete_comp.get_coeff_range(i);
            coeffs.assign(coeff_range.first, coeff_range.second);
            
            mask_comp.add_cell(complete_comp.get_label(i), complete_comp.get_dim(i), facets, coeffs);
        }
    }
       
    mask_comp.construct_cofacets();
    mask_comp.make_compressed(); 
        
    return mask_comp;
    
}


CellComplex construct_hypercube_complex(int dim, bool verbose=false) {
    
    CellComplex comp(dim);
    
   int ncells = 0;
        
    // Map of cell dimension to map of cell vertex list to cell index
    std::vector<std::map<std::vector<int>, int> > verts_to_cell(dim+2, std::map<std::vector<int>, int>());
    
    for(int k = 0; k <= dim; k++) {
        
        if(verbose) {
            py::print("dimension:", k, py::arg("flush")=true);
        }
        
        // For dimension k cells, iterate through every combination of k planar coordinates
        std::vector<bool> plane_mask(k, true);
        plane_mask.resize(dim, false);
        do { 
        
            std::vector<int> plane_coords;
            std::vector<int> perp_coords;
            for(int m = 0; m < dim; m++) {
                if(plane_mask[m]) {
                    plane_coords.push_back(m);
                } else {
                    perp_coords.push_back(m);
                }
            }


            // Iterate through all 2^(dim-k) possible combinations of perpendicular offsets
            for(int p1 = 0; p1 < (int) pow(2, dim-k); p1++) {

                // Turn on coords of offset
                std::vector<bool> coords(dim, false);
                for(int q = 0; q < dim-k; q++) {
                    // Check if qth bit is on
                    coords[perp_coords[q]] = (p1 & (1 << q));
                }
                                
                // Find all 2k facets in offset plane that comprise cell
                std::vector<int> facets;
                
                // Choose one extra direction that each facet is perpendicular to
                // Each facet is perpendicular to all the same directions as the cell (perp_coords)
                // But also this additional direction
                for(auto facet_perp_coord: plane_coords) {
                    
                    
                    std::vector<int> facet_plane_coords = plane_coords;
                    facet_plane_coords.erase(
                        std::remove(facet_plane_coords.begin(), facet_plane_coords.end(), facet_perp_coord),
                              facet_plane_coords.end());
                    
                    std::vector<int> facet_verts1;
                    std::vector<int> facet_verts2;
                    // Find all 2^(k-1) vertices in the offset plane that comprise cell 
                    for(int p2 = 0; p2 < (int) pow(2, k-1); p2++) {
                        
                        // Check if qth bit is on
                        for(int q = 0; q < k-1; q++) {
                            coords[facet_plane_coords[q]] = (p2 & (1 << q));
                        }
                        
                        //First turn perp direction off
                        coords[facet_perp_coord] = false;
                        
                        int vlabel = std::accumulate(coords.rbegin(), coords.rend(), 0, 
                                                 [](int x, int y) { return (x << 1) + y; });
                        
                        facet_verts1.push_back(vlabel);
                        
                        //Next turn perp direction on
                        coords[facet_perp_coord] = true;
                        
                        vlabel = std::accumulate(coords.rbegin(), coords.rend(), 0, 
                                                 [](int x, int y) { return (x << 1) + y; });
                        
                        facet_verts2.push_back(vlabel);
      
                    }
                    
                    facets.push_back(verts_to_cell[k][facet_verts1]);
                    facets.push_back(verts_to_cell[k][facet_verts2]);
                    
                }
                
                
                int label = verts_to_cell[k+1].size();
                
                std::vector<int> coeffs;


                comp.add_cell(label, k, facets, coeffs);

                
                
                // List of vertices comprising cell
                std::vector<int> verts;
                          
                // Go through every subset of coordinates and add vertex
                for(int p2 = 0; p2 < (int) pow(2, k); p2++) {
                    
                    // Check if qth bit is on
                    for(int q = 0; q < k; q++) {
                        coords[plane_coords[q]] = (p2 & (1 << q));
                    }
                    
                    int vlabel = std::accumulate(coords.rbegin(), coords.rend(), 0, 
                                                 [](int x, int y) { return (x << 1) + y; });
                    
                    
                    verts.push_back(vlabel);
                    
                }
                                
                // Store list of cells
                verts_to_cell[k+1][verts] = ncells;
                ncells++;
                
            }
            
        } while(std::prev_permutation(plane_mask.begin(), plane_mask.end()));
            
    }
    comp.construct_cofacets();
    comp.make_compressed(); 
        
    return comp;
    
}


// CellComplex construct_masked_cubical_complex(std::vector<bool> &mask, std::vector<int> &shape, bool oriented, bool dual) {
      
//     CellComplex complete_comp = construct_cubical_complex(shape, oriented, dual);
    
//     std::vector<bool> include(complete_comp.ncells, false);
    
//     for(int i = 0; i < complete_comp.ncells; i++) {
        
//         if((dual && complete_comp.get_dim(i) == complete_comp.dim) 
//           || (!dual && complete_comp.get_dim(i) == 0)) {
            
//             include[i] = !mask[complete_comp.get_label(i)];
//         }
//     }
    
//     if(dual) {
//         for(int d = complete_comp.dim-1; d >= 0; d--) {
//             for(int i = 0; i < complete_comp.ncells; i++) {
//                 if(complete_comp.get_dim(i) == d) {
                                        
//                     auto range = complete_comp.get_cofacet_range(i);
//                     for(auto it = range.first; it != range.second; it++) {
                        
//                         if(include[*it]) {
//                             include[i] = true;
//                         }
                        
//                     }
//                 }
//             }
//         }
//     } else {
//         for(int d = 1; d <= complete_comp.dim; d++) {
//             for(int i = 0; i < complete_comp.ncells; i++) {
//                 if(complete_comp.get_dim(i) == d) {
                                        
//                     auto range = complete_comp.get_facet_range(i);
//                     for(auto it = range.first; it != range.second; it++) {
                        
//                         if(include[*it]) {
//                             mask[i] = true;
//                         }
                        
//                     }
//                 }
//             }
//         }
//     }
      
    
//     std::vector<int> complete_to_mask(complete_comp.ncells);
//     int index = 0;
//     for(int i = 0; i < complete_comp.ncells; i++) {
//         if(include[i]) {
//             complete_to_mask[i] = index;
//             index++;
//         }
//     }
    
//     CellComplex mask_comp(complete_comp.dim, true, oriented);
    
//     for(int i = 0; i < complete_comp.ncells; i++) {
//         if(include[i]) {
            
//             std::vector<int> facets;
//             auto facet_range = complete_comp.get_facet_range(i);
//             for(auto it = facet_range.first; it != facet_range.second; it++) {
//                 facets.push_back(complete_to_mask[*it]);
//             }
//             std::vector<int> coeffs;
//             auto coeff_range = complete_comp.get_coeff_range(i);
//             coeffs.assign(coeff_range.first, coeff_range.second);
            
//             mask_comp.add_cell(complete_comp.get_label(i), complete_comp.get_dim(i), facets, coeffs);
//         }
//     }
       
//     mask_comp.construct_cofacets();
//     mask_comp.make_compressed(); 
        
//     return mask_comp;
    
// }


std::unordered_set<int> get_boundary_pixels(std::unordered_set<int> &pixels, std::vector<int> &shape) {
 
    std::unordered_set<int> boundary;
        
    int nrows = shape[0];
    int ncols = shape[1];
    
    for(auto pix: pixels) {
        int col = pix % ncols;
        int row = (pix - col) / ncols;
                
        if(row == 0 || row == nrows-1 || col == 0 || col == ncols-1) {
            boundary.insert(pix);
        } else if(!pixels.count(ncols*(row-1)+col) || !pixels.count(ncols*(row-1)+col+1) 
                 || !pixels.count(ncols*row+col+1) || !pixels.count(ncols*(row+1)+col+1)
                 || !pixels.count(ncols*(row+1)+col) || !pixels.count(ncols*(row+1)+col-1)
                 || !pixels.count(ncols*row+col-1) || !pixels.count(ncols*(row-1)+col-1)) {
            boundary.insert(pix);
        }
    }
    
    return boundary;
}

double calc_elongation(std::unordered_set<int> &pixels, std::vector<int> &shape) {
    
    int dim = shape.size();
    
    // int nrows = shape[0];
    int ncols = shape[1];
    
    XVec CM = XVec::Zero(dim);
    for(auto p: pixels) {
        int row = (p - p % ncols) / ncols;
        int col = p % ncols;
        
        CM(0) += row;
        CM(1) += col;
 
    }
    
    CM /= pixels.size();
    
    XMat I = XMat::Zero(dim, dim);
    
    for(auto p: pixels) {
        int row = (p - p % ncols) / ncols;
        int col = p % ncols;
        
        XVec x = XVec::Zero(dim);
        x(0) = row;
        x(1) = col;
        
        x -= CM;
        
        I += x.squaredNorm() * XMat::Identity(dim, dim) - x * x.transpose();
        
    }
    
    Eigen::SelfAdjointEigenSolver<XMat > eigensolver(I);
    XVec evals = eigensolver.eigenvalues();
    
    return sqrt(evals(0) / evals(1));
    
    
}

    
#endif // CUBICALCOMPLEX_HPP