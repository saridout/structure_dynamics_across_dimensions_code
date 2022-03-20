#ifndef MORSE_HPP
#define MORSE_HPP

#include "eigen_macros.hpp"
#include "cell_complex.hpp"
#include "filtration.hpp"
   
#include <pybind11/pybind11.h>    
namespace py = pybind11;
    
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <numeric>
#include <utility>
#include <math.h>
#include <time.h>
    

// Return lower star and upper costar of cell alpha
// alpha will be in both
std::tuple<std::unordered_set<int>, std::unordered_set<int> > get_star_decomp(int alpha, Filtration &filt, CellComplex &comp, int target_dim=-1) {
    
    std::unordered_set<int> faces = comp.get_faces(alpha, target_dim);
    std::unordered_set<int> cofaces = comp.get_cofaces(alpha, target_dim);
    
    int digi_func = filt.get_digi_func(alpha);
    int alpha_dim = comp.get_dim(alpha);
    
    std::unordered_set<int> lstar;
    std::unordered_set<int> ucostar;
    
    for(auto beta: faces) {
        if(filt.get_digi_func(beta) == digi_func) {
            if(comp.get_dim(beta) >= alpha_dim) {
                lstar.insert(beta);
            } 
            if (comp.get_dim(beta) <= alpha_dim) {
                ucostar.insert(beta);
            }
        }   
    }
    
    for(auto beta: cofaces) {
        if(filt.get_digi_func(beta) == digi_func) {
            if(comp.get_dim(beta) >= alpha_dim) {
                lstar.insert(beta);
            } 
            if (comp.get_dim(beta) <= alpha_dim) {
                ucostar.insert(beta);
            }
        }   
    }
    
    
    return std::make_pair(lstar, ucostar);
    
}


std::tuple<XiVec, XiVec>  construct_discrete_gradient(Filtration &filt, CellComplex &comp) {     
        
    auto lstar_cmp = [&filt, &comp](const int& lhs, const int &rhs) {
                
        // Always pop min value
        // But priority queues will default to popping largest value
        // for default < operator, so switch
        return filt(rhs, lhs);
        
    };
            
            
    auto ucostar_cmp = [&filt, &comp](const int& lhs, const int &rhs) {
        
        // Always pop max value
        // But priority queues will default to popping largest value
        // for default < operator, so no need to switch
        return filt(lhs, rhs);
        
    };
        
    XiVec V = XiVec::Constant(comp.ncells, -1);
    XiVec coV = XiVec::Constant(comp.ncells, -1);
    
    for(int c = 0; c < comp.ncells; c++) {
        
        // Skip cells that do not define the filtration
        if(filt.filt_dim != -1 && comp.get_dim(c) != filt.filt_dim) {
            continue;
        }
        
        // Get lower star and upper costar
        std::unordered_set<int> lstar;
        std::unordered_set<int> ucostar;
        std::tie(lstar, ucostar) = get_star_decomp(c, filt, comp);
                
        // First process upper costar portion of cells
        if(filt.filt_dim > 0 && ucostar.size() > 1) {
            // map of each cell to unpaired cofacets in ucostar
            std::unordered_map<int, std::unordered_set<int> > unpaired;
            for(auto alpha: ucostar) {
                auto range = comp.get_cofacet_range(alpha);
                for(auto it = range.first; it != range.second; it++) {
                    int beta = *it;
                    if(ucostar.count(beta)) {
                        unpaired[alpha].insert(beta);
                    }
                }

            }
                        
            // Cells with zero unpaired cofacets
            std::priority_queue<int, std::vector<int>, decltype(ucostar_cmp)> PQone(ucostar_cmp);
            // Cells with exactly one unpaired cofacet
            std::priority_queue<int, std::vector<int>, decltype(ucostar_cmp)> PQzero(ucostar_cmp);
            // Note: priority_queue's always pop the largest element first
            // (for deault < comparison operator)
            
            for(auto alpha: ucostar) {
                // Find intial set of cells with zero unpaired cofacets
                if(unpaired[alpha].empty()) {
                    PQzero.push(alpha);
                // Find initial set of cells with exactly one unpaired cofacet
                } else if(unpaired[alpha].size() == 1) {
                    PQone.push(alpha);
                }
            }
            
            // Identify all pairs and critical cells
            while(!PQone.empty() || !PQzero.empty()) {
            
                // First process all cells with exactly one unpaired cofacet
                while(!PQone.empty()) {
                    int alpha = PQone.top();
                    PQone.pop();

                    // If alpha doesn't actually have any unpaired cofacets, then add to PQzero
                    if(unpaired[alpha].empty()) {
                        PQzero.push(alpha);
                        continue;
                    }

                    // Unpaired cofacet of alpha
                    int beta = *(unpaired[alpha].begin());
                    
                    // alpha points to cofacet beta
                    V[alpha] = beta;
                    coV[beta] = alpha;

                    // alpha and beta are no longer unpaired, so remove from unpaired
                    unpaired.erase(alpha);
                    unpaired.erase(beta);
                    
                    for(auto kv: unpaired) {
                        
                        int gamma = kv.first;
                        
                        // Loop through facets of alpha and beta
                        if(unpaired[gamma].count(alpha) || unpaired[gamma].count(beta)) {
                            // Remove alpha and beta from list of unpaired
                            unpaired[gamma].erase(alpha);
                            unpaired[gamma].erase(beta);

                            // Add to PQOne if gamma now has only one unpaired cofacet
                            if(unpaired[gamma].size() == 1) {
                                PQone.push(gamma);
                            }
                        }
                    }
                }

                // Next process first cell with zero unpaired cofacets
                if(!PQzero.empty()) {
                    int alpha = PQzero.top();
                    PQzero.pop();

                    // If alpha is unpaired, then is critical
                    if(unpaired.count(alpha)) {
                        unpaired.erase(alpha);

                        for(auto kv: unpaired) {
                            
                            int gamma = kv.first;
                            
                            // Loop through facets of alpha
                            if(unpaired[gamma].count(alpha)) {
                                unpaired[gamma].erase(alpha);

                                // Add to PQOne if gamma now has only one unpaired cofacet
                                if(unpaired[gamma].size() == 1) {
                                    PQone.push(gamma);
                                }
                            }
                        }
                    }
                }
                
            }

        // Second process lower star portion of cells
        } else if(filt.filt_dim <= comp.dim && lstar.size() > 1) {
                
            // map of each cell to unpaired facets in lstar
            std::unordered_map<int, std::unordered_set<int> > unpaired;
            for(auto alpha: lstar) {
                
                // skip if already paired during lstar
                if(V[alpha] != -1 || coV[alpha] != -1) {
                    continue;
                }
                
                auto range = comp.get_facet_range(alpha);
                for(auto it = range.first; it != range.second; it++) {
                    int beta = *it;
                    
                    // skip if already paired during lstar
                    if(lstar.count(beta) && V[beta] == -1 && coV[beta] == -1) {
                        unpaired[alpha].insert(beta);
                    }
                }

            }
            
            // Cells with zero unpaired facets
            std::priority_queue<int, std::vector<int>, decltype(lstar_cmp)> PQone(lstar_cmp);
            // Cells with exactly one unpaired facet
            std::priority_queue<int, std::vector<int>, decltype(lstar_cmp)> PQzero(lstar_cmp);
            // Note: priority_queue's always pop the largest element first
            // (for deault < comparison operator)
            
            for(auto alpha: lstar) {
                // Find intial set of cells with zero unpaired facets
                if(unpaired[alpha].empty()) {
                    PQzero.push(alpha);
                // Find initial set of cells with exactly one unpaired facet
                } else if(unpaired[alpha].size() == 1) {
                    PQone.push(alpha);
                }
            }
            
            // Identify all pairs and critical cells
            while(!PQone.empty() || !PQzero.empty()) {
            
                // First process all cells with exactly one unpaired facet
                while(!PQone.empty()) {
                    int alpha = PQone.top();
                    PQone.pop();

                    // If alpha doesn't actually have any unpaired facets, then add to PQzero
                    if(unpaired[alpha].empty()) {
                        PQzero.push(alpha);
                        continue;
                    }

                    // Unpaired facet of alpha
                    int beta = *(unpaired[alpha].begin());
                    
                    // beta points to cofacet alpha
                    V[beta] = alpha;
                    coV[alpha] = beta;

                    // alpha and beta are no longer unpaired, so remove from unpaired
                    unpaired.erase(alpha);
                    unpaired.erase(beta);
                    
                    for(auto kv: unpaired) {
                        
                        int gamma = kv.first;
                        
                        // Loop through cofacets of alpha and beta
                        if(unpaired[gamma].count(alpha) || unpaired[gamma].count(beta)) {
                            // Remove alpha and beta from list of unpaired
                            unpaired[gamma].erase(alpha);
                            unpaired[gamma].erase(beta);

                            // Add to PQOne if gamma now has only one unpaired facet
                            if(unpaired[gamma].size() == 1) {
                                PQone.push(gamma);
                            }
                        }
                    }
                }

                // Next process all cells with zero unpaired facets
                if(!PQzero.empty()) {
                    int alpha = PQzero.top();
                    PQzero.pop();

                    // If alpha is unpaired, then is critical
                    if(unpaired.count(alpha)) {
                        unpaired.erase(alpha);

                        for(auto kv: unpaired) {
                            
                            int gamma = kv.first;
                            
                            // Loop through cofacets of alpha
                            if(unpaired[gamma].count(alpha)) {
                                unpaired[gamma].erase(alpha);

                                // Add to PQOne if gamma now has only one unpaired facet
                                if(unpaired[gamma].size() == 1) {
                                    PQone.push(gamma);
                                }
                            }
                        }
                    }
                }

            }
            
        }
        
        
        // Finally record any remaining critical cells
        
        for(auto alpha: ucostar) {
            if(V[alpha] == -1 && coV[alpha] == -1) {
                V[alpha] = alpha;
                coV[alpha] = alpha;
            }
        }
        
        for(auto alpha: lstar) {
            if(V[alpha] == -1 && coV[alpha] == -1) {
                V[alpha] = alpha;
                coV[alpha] = alpha;
            }
        }
        
        
            
        
    }
    
    return std::make_tuple(V, coV);
    
}

// co determines whether search with V or coV
std::vector<std::tuple<int, int, int> > traverse_flow(int s, RXiVec V, 
                                                      CellComplex &comp, bool co=false, bool coordinated=false) {
    
    
    std::unordered_map<int, int> nrIn;
    std::unordered_map<int, int> k;
    
    if(coordinated) {
        std::vector<std::tuple<int, int, int> > traversal = traverse_flow(s, V, comp, co, false);
        for(auto trip: traversal) {
            int c = std::get<2>(trip);
            nrIn[c] += 1;
        }
    }
    
    std::queue<int> Q;
    Q.push(s);
    
    std::unordered_set<int> seen;
    seen.insert(s);
    
    std::vector<std::tuple<int, int, int> > traversal;
    
    while(!Q.empty()) {
        int a = Q.front();
        Q.pop();
                
        auto range = co ? comp.get_cofacet_range(a) : comp.get_facet_range(a);
        for(auto it = range.first; it != range.second; it++) {
            int b = *it;
                        
            if(V[b] != -1 && V[b] != a) {
                int c = V[b];
                traversal.emplace_back(a, b, c);
                
                if(!seen.count(c) && b != c) {
                    if(coordinated) {
                        k[c] += 1;
                        
                        if(k[c] != nrIn[c]) {
                            continue;
                        }
                        
                    }
                    
                    
                    Q.push(c);
                    seen.insert(c);
                    
                }   
            }
        }
        
    }
    
    return traversal;
    
}


std::vector<std::tuple<int, int, int> > find_morse_boundary(int s, RXiVec V, 
                                                             CellComplex &comp, bool co=false, bool oriented=false) {
    
    std::unordered_map<int, int> counts;
    counts[s] = 1;
    
    std::unordered_set<int> boundary;
    
    std::unordered_map<int, int> mult;
    if(oriented) {
        mult[s] = 1;
    }
    
    std::vector<std::tuple<int, int, int> > traversal = traverse_flow(s, V, comp, co, true);
    for(auto trip: traversal) {
        int a, b, c;
        std::tie(a, b, c) = trip;
        
        counts[c] += counts[a];
        
        if(b == c) {
            boundary.insert(c);
            
            // if oriented:
//                 mult[c] = mult.get(c, 0) + mult[a] * ( -coeff(a)[b] )
            
//         elif oriented:
//             mult[c] = mult.get(c, 0) + mult[a] * ( -coeff(a)[b] * coeff(c)[b] )
        }
    }
    
    std::vector<std::tuple<int, int, int> > morse_boundary;
    for(auto c: boundary) {
        if(oriented) {
            morse_boundary.emplace_back(c, counts[c], mult[c]);
        } else {
            morse_boundary.emplace_back(c, counts[c], counts[c] % 2);
        }
    }
    
    return morse_boundary;
    
}



CellComplex construct_morse_complex(RXiVec V, CellComplex &comp, bool oriented=false) {
    
    
    CellComplex morse_comp(comp.dim, false, oriented);
    
    
    // Morse complex is labeled according to corresponding cell in original complex
    // To get label of cell, consult original complex
    std::unordered_map<int, int> label_to_cell;
    std::vector<int> cell_to_label;
    
    int index = 0;
    for(int s = 0; s < comp.ncells; s++) {
        if(V[s] == s) {
            
            label_to_cell[s] = index;
            cell_to_label.push_back(s);
            
            index++;
            
        }
    }

    
    for(auto s: cell_to_label) {
        
        std::vector<int> facets;
        std::vector<int> coeffs;

        std::vector<std::tuple<int, int, int> > morse_boundary = find_morse_boundary(s, V, comp, false, oriented);
        for(auto trip: morse_boundary) {
            int c, k, m;
            std::tie(c, k, m) = trip;

            facets.push_back(label_to_cell[c]);
            coeffs.push_back(m);
            
            if(m > 1 || m < -1) {
                py::print("Large Coefficient:", k, m, comp.get_dim(c), comp.get_dim(s));
            }
        }
        
        morse_comp.add_cell(s, comp.get_dim(s), facets, coeffs);
    }
    
    morse_comp.construct_cofacets();
    morse_comp.make_compressed();
    return morse_comp;
    
}


Filtration construct_morse_filtration(Filtration &filt, CellComplex &morse_comp) {
    
    
    XVec func =  XVec::Zero(morse_comp.ncells);  
    XiVec digi_func = XiVec::Zero(morse_comp.ncells);
    XiVec order = XiVec::Zero(morse_comp.ncells);
    
    for(int c = 0; c < morse_comp.ncells; c++) {
        func(c) = filt.get_func(morse_comp.get_label(c));
        digi_func(c) = filt.get_digi_func(morse_comp.get_label(c));
        order(c) = filt.get_order(morse_comp.get_label(c));
    }
    
    return Filtration(morse_comp, func, digi_func, order, true);
    
}


std::vector<std::tuple<int, int, int> > find_connections(int s, int t, RXiVec V,  RXiVec coV, CellComplex &comp) {
    
    std::unordered_set<int> active;
    active.insert(t);
        
    std::vector<std::tuple<int, int, int> > traversal = traverse_flow(t, coV, comp, true, false);
    for(auto trip: traversal) {
        int c = std::get<2>(trip);
        active.insert(c);
    }
        
    std::vector<std::tuple<int, int, int> > connections;
    
    traversal = traverse_flow(s, V, comp, false, false);
    for(auto trip: traversal) {
        int b = std::get<1>(trip);
        if(active.count(b)) {
            connections.push_back(trip);
        }
    }
        
    return connections;
    
}




// Convert cell alpha in morse complex to representation in original cell complex
// co determines V or coV
std::unordered_set<int> find_morse_feature(int s, RXiVec V, CellComplex &comp, bool co=false) {
    
    
    std::unordered_set<int> feature;
    feature.insert(s);
    
    // Find rest of cells in original complex that are represented by cell s
    std::vector<std::tuple<int, int, int> > traversal = traverse_flow(s, V, comp, co, false);
    for(auto trip: traversal) {
        int b, c;
        std::tie(std::ignore, b, c) = trip;
                
        if(b != c) {
            feature.insert(c);
        }
    }
    
    return feature;
    
}

std::unordered_map<int, std::unordered_set<int> > find_morse_cells(std::vector<int> &cells, RXiVec V, 
                                                  CellComplex &mcomp, CellComplex &comp, bool co=false) {
    
    
    std::unordered_map<int, std::unordered_set<int> > mcells;
    
    for(auto c: cells) {
        
        auto feature = find_morse_feature(mcomp.get_label(c), V, comp, co);
        
        mcells.emplace(std::piecewise_construct, std::forward_as_tuple(mcomp.get_label(c)), 
                           std::forward_as_tuple(feature.begin(), feature.end()));
        
    }
    
    
    return mcells;
    
}


std::unordered_map<int, std::unordered_set<int> > find_morse_features(std::vector<int> &cells, RXiVec V, 
                                                  CellComplex &comp, bool co=false) {
    
    
    std::unordered_map<int, std::unordered_set<int> > features;
    
    for(auto c: cells) {
        
        auto feature = find_morse_feature(c, V, comp, co);
        
        features.emplace(std::piecewise_construct, std::forward_as_tuple(c), 
                           std::forward_as_tuple(feature.begin(), feature.end()));
        
    }
    
    
    return features;
    
}

// Returns map of vertices in morse complex to collections of vertices comprising basins in original complex
std::unordered_map<int, std::unordered_set<int> > find_morse_basins(RXiVec coV, CellComplex &morse_comp, CellComplex &comp) {
    
    std::vector<int> cells;
    
    for(int c = 0; c < morse_comp.ncells; c++) {
        
        if(morse_comp.get_dim(c) == 0) {
            cells.push_back(morse_comp.get_label(c));
        }
        
    }
    
    return find_morse_features(cells, coV, comp, true);
        
}




// Finds the morse skeleton of dimension skeleton_dim comprised of cells in the original complex
std::unordered_set<int> find_morse_skeleton(RXiVec V, CellComplex &morse_comp, CellComplex &comp, int skeleton_dim=1) {
        
    std::unordered_set<int> skeleton;
    
    for(int c = 0; c < morse_comp.ncells; c++) {
        
        if(morse_comp.get_dim(c) != skeleton_dim) {
            continue;
        }
        
        auto feature = find_morse_feature(morse_comp.get_label(c), V, comp);
        
        
        skeleton.insert(feature.begin(), feature.end());
        
    }
    
    
    return skeleton;
}

// Sorts vertices into voids determined by morse skeleton of highest dimension
std::unordered_map<int, std::unordered_set<int> > 
    convert_morse_voids_to_basins(std::unordered_map<int, std::unordered_set<int> > &mvoids,
                                                                   CellComplex &comp, Filtration &filt) {
 
    std::unordered_map<int, std::unordered_set<int> > basins;
    std::unordered_map<int, int> cells_to_voids;
    for(auto& v: mvoids) {
        
        int mvi = v.first;
        
//         basins[mvi];
        
        for(int c: v.second) {
            cells_to_voids[c] = mvi;
        }
    }
    
    auto cmp = [&filt] (const int &lhs, const int &rhs) {
        return filt(rhs, lhs);
    };
    
    std::unordered_set<int> unsorted;
    
    // Assign vertices to latest coface
    for(int vi = comp.dcell_range[0].first; vi < comp.dcell_range[0].second; vi++) {

        auto cofaces = comp.get_cofaces(vi, comp.dim);
//         std::vector<int> coface_list(cofaces.begin(), cofaces.end());
        
//         std::sort(coface_list.begin(), coface_list.end(), cmp);
        
//         bool inserted = false;
//         for(int cf: coface_list) {
            
//             if(cells_to_voids.count(cf)) {
//                 basins[cells_to_voids[cf]].insert(vi);
//                 inserted = true;
//                 break;
//             }  
//         }
        
//         if(!inserted) {
//             unsorted.insert(vi);
//         }
        
        int imax = *std::min_element(cofaces.begin(), cofaces.end(), cmp);
        
        if(cells_to_voids.count(imax)) {
            cells_to_voids[vi] = cells_to_voids[imax];
            basins[cells_to_voids[imax]].insert(vi);
        } else {
            unsorted.insert(vi);
        }
        
    }
    
    
    for(int vi : unsorted) {
        
        std::unordered_set<int> neighbors;
        
        for(int ei: comp.get_cofacets(vi)) {
            
            auto verts = comp.get_facets(ei);
            
            neighbors.insert(verts.begin(), verts.end());
            
        }
        
        neighbors.erase(vi);
        
        
        std::vector<int> neighbor_list(neighbors.begin(), neighbors.end());
        std::sort(neighbor_list.begin(), neighbor_list.end(), cmp);
        
        bool inserted = false;
        for(int vj: neighbor_list) {
            if(cells_to_voids.count(vj)) {
                basins[cells_to_voids[vj]].insert(vi);
                
                inserted = true;
                break;
            }
        }
        
        if(!inserted) {
            py::print("Unsorted", vi);
        }
        
    }
    
    
    
    return basins;
    
}

/***************************************************

If inclusive = false:

Converts cells of one dimension to cells of target_dim that have the same value of the digitized function
That is, they belong to the same lower star or upper costar

If inclusive = true:

Some cells are impossible to reach in this way, that is, each lower star or upper costar 
does not necessarily have cells of all possible dimensions

Sometimes, we want to map all cells of one dimension to all cells of another,

To accomplish this, set inclusive=true

Ex. Convert vertices to 2-cells (w/ filtration defined on 2-cells)

Each 2-cell should map to exactly one vertex
But each vertex can map to none or multiple 2-cells
w/o overlaps between vertices

Each 2-cell maps to lowest vertex in inclusion set
or if no vertices, then just to lowest boundary vertex

1. For each vertex find all 2-cells that are cofaces
2. For each 2-cell, find vertices in inclusion set
3. If nonzero vertices in inclusion set, check if current vertex is lowest (this avoids overlaps)
4. Otherwise check if current vertex is lowest face

This method ensure that vertices corresponding to minima always map to 2-cell


Ex. Convert 2-cells to vertices (w/ filtration defined on vertices)

Each vertex should map to exactly one 2-cell
But each 2-cell can map to none or multiple vertices
w/o overlaps between 2-cell

Each vertex maps to highest 2-cell in inclusion set
or if no 2-cells, then just to highest coboundary 2-cell

1. For each 2-cells find all vertices that are faces
2. For each vertex, find 2-cells in inclusion set
3. If nonzero 2-cells in inclusion set, check if current 2-cell is highest (this avoids overlaps)
4. Otherwise check if current 2-cell is highest coface

This method ensure that 2-cells corresponding to maxima always map to a vertex


***************************************************/
std::unordered_set<int> convert_feature_dim(std::unordered_set<int> &feature, 
                                            int target_dim, Filtration &filt, CellComplex &comp, bool inclusive=true) {
        
    std::unordered_set<int> new_feature;
    
    
    for(auto s: feature) {
        
        if(inclusive) {
        
        
            // If cell dim is higher than target dim
            if(comp.get_dim(s) > target_dim) {

                // Get faces of cell that are of target_dim
                auto faces = comp.get_faces(s, target_dim);

                // Iterate through each face
                for(auto f: faces) {

                    // Find all cofaces of face
                    auto cofaces = comp.get_cofaces(f, comp.get_dim(s));
                    
                    // Find all cofaces in same lower star as f
                    std::unordered_set<int> lstar;
                    for(auto a: cofaces) {
                        if(filt.get_digi_func(f) == filt.get_digi_func(a)) {
                            lstar.insert(a);
                        }
                    }
                    
                    // If lstar is not empty and s is the maximum in lstar
                    // Or lstar is empty but s is still the maximum
                    if( (!lstar.empty() && (s == *std::max_element(lstar.begin(), lstar.end(), filt)))
                      || (lstar.empty() && (s == *std::max_element(cofaces.begin(), cofaces.end(), filt))) ) {
                        new_feature.insert(f);
                    }
                }

                
            // If cell dim is lower than target dim
            } else {
                    
                // Get cofaces of cell that are of target_dim
                auto cofaces = comp.get_cofaces(s, target_dim);

                // Iterate through each face
                for(auto cf: cofaces) {

                    // Find all faces of coface
                    auto faces = comp.get_faces(cf, comp.get_dim(s));
                    
                    // Find all faces in same upper costar as cf
                    std::unordered_set<int> ucostar;
                    for(auto a: faces) {
                        if(filt.get_digi_func(cf) == filt.get_digi_func(a)) {
                            ucostar.insert(a);
                        }
                    }
                    
                    // If ucostar is not empty and s is the minimum in ucostar
                    // Or ucostar is empty but s is still the minimum
                    if( (!ucostar.empty() && (s == *std::min_element(ucostar.begin(), ucostar.end(), filt)))
                      || (ucostar.empty() && (s == *std::min_element(faces.begin(), faces.end(), filt))) ) {
                        new_feature.insert(cf);
                    }
                }
                    
            }
            
        } else {
            
            std::unordered_set<int> lstar;
            std::unordered_set<int> ucostar;

            std::tie(lstar, ucostar) = get_star_decomp(s, filt, comp, target_dim);
            
            new_feature.insert(lstar.begin(), lstar.end());
            new_feature.insert(ucostar.begin(), ucostar.end());
        
        }
    }
    
    return new_feature;
    
}




// // Update this to look more like extract_persistence_feature
// // Also change so that it immediately spits out morse feature by default
// std::unordered_set<int> extract_morse_feature(int i, int j, CellComplex &mcomp, StarFiltration &filt, int target_dim=-1, bool complement=false) {
        
    
//     bool co = (mcomp.get_dim(i) != 0);
    
//     if(target_dim == -1) {
//         target_dim = co ? mcomp.dim : 0;
//     }
    
//     std::unordered_set<int> seen;
//     std::queue<int> Q;
//     if(!co) {
//         seen.insert(i);
//         Q.push(i);
//     } else {
//         seen.insert(j);
//         Q.push(j);
//     }
  
//     int orderi = filt.get_total_order(mcomp.get_label(i));
//     int orderj = filt.get_total_order(mcomp.get_label(j));
         
//     while(!Q.empty()) {
//         int a = Q.front();
//         Q.pop();
        
//         // py::print("a", a);
        
//         for(auto b: get_star(a, co, mcomp, -1)) {
            
//             // py::print("b", b);
                        
//             if((!co && filt.get_total_order(mcomp.get_label(b)) >= orderj)
//               || (co && filt.get_total_order(mcomp.get_label(b)) <= orderi)) {
//                 continue;
//             }

//             for(auto c: get_star(b, !co, mcomp, -1)) {
//                 // py::print("c", c);
                
//                 if((!co && filt.get_total_order(mcomp.get_label(c)) <= orderi)
//                   || (co && filt.get_total_order(mcomp.get_label(c)) >= orderj)) {
//                     continue;
//                 }
                
//                 if(!seen.count(c) && c != a) {
//                     Q.push(c);
//                     seen.insert(c);
//                 }
                
//             }
//         }
//     }
    
//     if(complement) {
        
//         std::unordered_set<int> comp_seen;
//         if(!co) {
//             seen.insert(j);
//             Q.push(j);
//         } else {
//             seen.insert(i);
//             Q.push(i);
//         }
        
//         while(!Q.empty()) {
//             int a = Q.front();
//             Q.pop();

//             // py::print("a", a);

//             for(auto b: get_star(a, !co, mcomp, -1)) {

//                 // py::print("b", b);

//                 for(auto c: get_star(b, co, mcomp, -1)) {
//                     // py::print("c", c);
                    
//                     if((!co && filt.get_total_order(mcomp.get_label(c)) >= orderj)
//                       || (co && filt.get_total_order(mcomp.get_label(c)) <= orderi)) {
//                         continue;
//                     }
                    
//                     if(!seen.count(c) && !comp_seen.count(c) && c != a) {
//                         Q.push(c);
//                         comp_seen.insert(c);
//                     }

//                 }
//             }
//         }
        
//         seen = comp_seen;
        
//     }
    
//     std::unordered_set<int> feature;
    
//     for(auto s: seen) {
//         if(mcomp.get_dim(s) == target_dim) {
//             feature.insert(mcomp.get_label(s));
//         }
//     }
    
//     return feature;
    
// }

    






// std::unordered_set<int> extract_morse_feature_to_real(int i, int j, CellComplex &mcomp, py::array_t<int> V, py::array_t<int> coV,
//                                                       CellComplex &comp, StarFiltration &filt, bool complement=false, int target_dim=-1) {
    
    
//     if(target_dim == -1) {
//         target_dim = filt.fdim;
//     }
    
//     std::unordered_set<int> feature = extract_morse_feature(i, j, mcomp, filt, -1, complement);
//     feature = convert_morse_to_real(feature, V, coV, comp);    
//     feature = change_feature_dim(feature, target_dim, filt, comp, true);
//     feature = comp.get_labels(feature);
    
//     return feature;
    
// }

// std::unordered_set<int> extract_morse_basin(int s, CellComplex &mcomp, py::array_t<int> V, py::array_t<int> coV,
//                                                       CellComplex &comp, StarFiltration &filt, int target_dim=-1) {
    
    
//     if(target_dim == -1) {
//         target_dim = filt.fdim;
//     }
    
//     std::unordered_set<int> mfeature;
//     mfeature.insert(mcomp.get_label(s));    
    
//     // Find the corresponding cells in real complex
//     std::unordered_set<int> rfeature = convert_morse_to_real(mfeature, V, coV, comp);

//     // Change dimension of features to representative dimension
//     std::unordered_set<int> feature = change_feature_dim(rfeature, target_dim, filt, comp, true);

//     std::unordered_set<int> feature_labels = comp.get_labels(feature);
    
//     return feature_labels;
    
// }


// /***************************************************
// Finds the morse basins and converts to target_dim

// ***************************************************/




// std::unordered_set<int> find_morse_basin_borders(CellComplex &mcomp, py::array_t<int> V, py::array_t<int> coV, 
//                                                                    StarFiltration &filt, CellComplex &comp, int target_dim=-1) {
        
//     std::unordered_set<int> basin_borders;
    
//     if(target_dim == -1) {
//         target_dim = filt.fdim;
//     }
        
//     for(int c = 0; c < mcomp.ncells; c++) {
//         if(mcomp.get_dim(c) == 0) {
//             // Cell in original complex
//             int s = mcomp.get_label(c);
            
//             std::unordered_set<int> mfeature;
//             mfeature.insert(s);
            
//             // Find the corresponding cells in real complex
//             std::unordered_set<int> rfeature = convert_morse_to_real(mfeature, V, coV, comp);
                               
//             // Change dimension of features to representative dimension
//             std::unordered_set<int> feature = change_feature_dim(rfeature, target_dim, filt, comp, true);
            
//             std::unordered_set<int> boundary = get_boundary(feature, comp);
            
//             std::unordered_set<int> border = change_feature_dim(boundary, target_dim, filt, comp, false);
            
//             std::unordered_set<int> feature_labels = comp.get_labels(border);

//             basin_borders.insert(feature_labels.begin(), feature_labels.end());
                     
//         }
//     }
    
//     return basin_borders;
// }
    



    
#endif // MORSE_HPP