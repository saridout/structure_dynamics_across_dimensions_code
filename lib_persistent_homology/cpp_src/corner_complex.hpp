#ifndef CORNERCOMPLEX_HPP
#define CORNERCOMPLEX_HPP
    
#include "eigen_macros.hpp"
#include "cell_complex.hpp"
#include "embedding.hpp"
#include "graph_complex.hpp"

#include <pybind11/pybind11.h>    
namespace py = pybind11;

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>



template <int DIM> std::vector<std::vector<int> > find_corners(Graph &graph, Embedding<DIM> &embed) {
    
    
    graph.construct_neighbor_list();
        
    std::vector<std::vector<int> > corners;
        
    // Iterate over each vertex
    for(int vi = 0; vi < graph.NV; vi++) {
                
        // Get all vertices
        std::vector<int> verts;
        // Get all vertex positions relative to center vertex
        std::unordered_map<int, DVec > positions;
        
        verts.push_back(vi);
        positions[vi] = DVec::Zero();
        
        // Center vertex positions
        DVec O = embed.get_vpos(vi);
            
        for(auto vj: graph.neighbors[vi]) {
            verts.push_back(vj);
            
            DVec bvec = embed.get_vpos(vj) - O;
            
            for(int d = 0; d < DIM; d++) {
                if(std::fabs(bvec(d)) > 0.5) {
                    bvec(d) -= ((bvec(d) > 0) - (bvec(d) < 0));
                }
            }

            bvec =  embed.box_mat * bvec;
                                  
            positions[vj] = bvec.normalized();
        }
                        
        // Must have at least dim vertices other than the central vertex for a corner to exist
        if((int)verts.size() < DIM + 1) {
            continue;
        }

        // A mask to pick out exactly dim verts
        std::vector<bool> mask(DIM, true);
        mask.resize(verts.size(), false);
        
        // Iterate through each simplex and test if part of convex hull
        std::vector<std::unordered_set<int> > hull_simps;
        do {  

            // Find vertices in simplex
            std::unordered_set<int> simplex;
            for(std::size_t j = 0; j < verts.size(); j++) {
                if(mask[j]) {
                    simplex.insert(verts[j]);
                }
            }
                                    
            // Calculate positions of all verts relative to one vert
            
            XMat cross_mat(DIM, DIM-1);
            
            auto sit = simplex.begin();
            DVec A = positions[*sit];
                        
            sit++;
            for(int m = 0; m < DIM-1; m++, sit++) {
                cross_mat.col(m) = positions[*sit] - A;
            }
                        
            // Use cofactors of cross product matrix to calculate normal vector
            DVec normal;
            for(int m = 0; m < DIM; m++) {
                
                XMat cofactor_mat(DIM-1, DIM-1);
                cofactor_mat.block(0, 0, m, DIM-1) = cross_mat.block(0, 0, m, DIM-1);
                cofactor_mat.block(m, 0, DIM-1-m, DIM-1) = cross_mat.block(m+1, 0, DIM-1-m, DIM-1);
                
                int sign = (m % 2 == 0) ? 1 : -1;
                normal(m) = sign * cofactor_mat.determinant();
            }
                        
            // Check if all points are colinear
            double norm = normal.norm();
            if(norm < 1e-8) {
                continue;
            }
            normal /= norm;
                        
            int above = 0;
            int below = 0;
            for(std::size_t j = 0; j < verts.size(); j++) {
                if(!mask[j]) {
                    
                    DVec v = positions[verts[j]] - A;
                    
                    if(v.dot(normal) > 0) {
                        above++;
                    } else {
                        below++;
                    }
                    
                }
            }
            
            if(above == 0 || below == 0) {
                hull_simps.push_back(simplex);
            }

            
        } while(std::prev_permutation(mask.begin(), mask.end()));
        
        for(auto simplex: hull_simps) {
            
            if(!simplex.count(vi)) {
                
                corners.emplace_back(1, vi);
                corners.back().insert(corners.back().end(), simplex.begin(), simplex.end());
                
                std::sort(corners.back().begin()+1, corners.back().end());
                
            }
            
        }
        
    }
    
    std::sort(corners.begin(), corners.end());
    
    return corners;
    
}



template <int DIM> XVec calc_corner_strains(std::vector< std::vector<int> > &corners, 
                                                           RXVec disp, Embedding<DIM> &embed, bool strain=false) {

    
    XVec corner_strains= XVec::Zero(corners.size());
        
    for(std::size_t i = 0; i < corners.size(); i++) {
        
        auto corner = corners[i];
        
        int vi = corner[0];
        
        DVec O = embed.get_vpos(vi);
        DVec uO = disp.segment<DIM>(DIM*vi);

        DMat X = DMat::Zero();
        DMat Y = DMat::Zero();
        
//         DVec bvec_avg = DVec::Zero();
        
        for(int m = 0; m < DIM; m++) {  

            int vj = corner[1+m];
            
            DVec posj = embed.get_vpos(vj);
            
            DVec bvec = embed.get_diff(O, posj);
            DVec du = disp.segment<DIM>(DIM*vj) - uO;
            
            X += bvec * bvec.transpose();
            Y += du * bvec.transpose();
            
//             bvec_avg += bvec;

        }
        
//         bvec_avg.normalize();

        
        DMat F = Y * X.inverse();
        
        DMat eps = 0.5 * (F + F.transpose());
                    
        double gamma = 0.0;
        
        if(strain) {
            gamma = eps.norm();
        } else {
            for(int m = 0; m < DIM; m++) {  

                int vj = corner[1+m];

                DVec posj = embed.get_vpos(vj);

                DVec bvec = embed.get_diff(O, posj);

                gamma += (eps*bvec).squaredNorm();

            }

            gamma = sqrt(gamma);
        }
        
//         DVec Ovec = embed.get_pos(vi) - 0.5*DVec::Ones();
        
// //         if(X.determinant() < 1e-4) {
//             py::print(i, "det", X.determinant(), Y.determinant(), eps.norm(), bvec_avg.dot(Ovec / Ovec.norm()), Ovec.norm());
            
//             py::print("X", X);
//             py::print("Y", Y);
//             py::print("eps", eps);
// //         }
        
        
        
        
        corner_strains[i] = gamma;
        
         
    }
    
    return corner_strains;
    
    
}

template <int DIM> std::tuple<XVec, XVec> calc_corner_flatness(std::vector< std::vector<int> > &corners, Embedding<DIM> &embed) {
    
    XVec min_eval = XVec::Zero(corners.size());
    XVec max_eval = XVec::Zero(corners.size());
    
    
    for(std::size_t i = 0; i < corners.size(); i++) {
        
        auto corner = corners[i];
        
        int vi = corner[0];
        
        DVec O = embed.get_vpos(vi);

        DMat X = DMat::Zero();
                
        for(int m = 0; m < DIM; m++) {  

            int vj = corner[1+m];
            
            DVec posj = embed.get_vpos(vj);
            
            DVec bvec = embed.get_diff(O, posj);
            bvec.normalize();
            
            X += bvec * bvec.transpose();
            
        }
        
        Eigen::SelfAdjointEigenSolver<DMat> esolver(X);
        
        DVec evals = esolver.eigenvalues();
                
        min_eval[i] = evals[0];
        max_eval[i] = evals[DIM-1];
        
    }
    
    return std::make_tuple(min_eval, max_eval);
    
}


// Creates cell complex from network with d-dimensional simplices at each "corner"
// Homotopically equivalent to orginal network
template <int DIM> CellComplex construct_corner_complex(std::vector<std::vector<int> > &corners, Graph &graph) {
     
    CellComplex comp(DIM, true, false);
          
    // For each vertex, map of lists of vertices representing each simplex in corner to index in full cell complex
    std::vector<std::map<std::vector<int> , int> > vertices_to_index(graph.NV);
    
    // First process vertices
    for(int i = 0; i < graph.NV; i++) {

        std::vector<int> facets;
        std::vector<int> coeffs;
        comp.add_cell(i, 0, facets, coeffs);
        
        vertices_to_index[i].emplace(std::piecewise_construct, std::forward_as_tuple(1, i), std::forward_as_tuple(i));
        
    }
    
    // Next process edges
    for(int i = 0; i < graph.NE; i++) {
        int vi = graph.edgei[i];
        int vj = graph.edgej[i];
        
        std::vector<int> facets;
        facets.push_back(vi);
        facets.push_back(vj);

        std::vector<int> coeffs;
        
        comp.add_cell(graph.NV + i, 1, facets, coeffs);
        
        std::sort(facets.begin(), facets.end());
        vertices_to_index[vi][facets] = graph.NV + i;
        vertices_to_index[vj][facets] = graph.NV + i;
        
        vertices_to_index[vi].emplace(std::piecewise_construct, std::forward_as_tuple(1, vj), std::forward_as_tuple(vj));
        vertices_to_index[vj].emplace(std::piecewise_construct, std::forward_as_tuple(1, vi), std::forward_as_tuple(vi));
    }
    
//     for(std::size_t vi = 0; vi < vertices_to_index.size(); vi++) {
//         py::print(vi);
        
//         for(auto pair: vertices_to_index[vi]) {
//             py::print(pair.first, pair.second);
//         }
        
//     }
    
    std::vector<int> corner_to_cell;
    
    // Iterate through each dimension up to corner dimension
    for(int d = 1; d <= DIM; d++) {
        
        // Iterate through each corner
        for(std::size_t i = 0; i < corners.size(); i++) {
            
            int vi = corners[i][0];
            
            // Iterate through every size d+1 subset
            
            // A mask to pick out exactly d+1 verts
            std::vector<bool> mask(d+1, true);
            mask.resize(corners[i].size(), false);
            do {  
                
                // Find vertices in simplex
                std::vector<int> simplex;
                for(std::size_t j = 0; j < corners[i].size(); j++) {
                    if(mask[j]) {
                        simplex.push_back(corners[i][j]);
                    }
                }
                
                // Sorted list of vertices of cell
                std::sort(simplex.begin(), simplex.end());
                
                
//                 py::print(vi, simplex);
                
                // If simplex already exists in graph complex, then skip                
                if(vertices_to_index[vi].count(simplex)) {
                    continue;
                }
                                
                vertices_to_index[vi][simplex] = comp.ncells;
                
                // Find facets
                std::vector<int> facets;
                std::vector<int> coeffs;
                for(std::size_t j = 0; j < simplex.size(); j++) {
                    std::vector<int> facet(simplex);
                    facet.erase(facet.begin()+j);
                    facets.push_back(vertices_to_index[vi][facet]);
                }
                
                // Label new cells -1 to indicate they don't correspond to anything in original graph complex
                // Or if cell corresponds to corner, label it with corner index
                if(d == DIM) {
                    comp.add_cell(i, d, facets, coeffs);
                } else {
                    comp.add_cell(-1, d, facets, coeffs);
                }
                    
                
            } while(std::prev_permutation(mask.begin(), mask.end()));

        }
        
    }
    
    
    comp.construct_cofacets();
    comp.make_compressed(); 

    return comp;
}

        
    

    
#endif // CORNERCOMPLEX_HPP