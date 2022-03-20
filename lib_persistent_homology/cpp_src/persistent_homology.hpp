#ifndef PERSIST_HPP
#define PERSIST_HPP
    
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <math.h>
    
#include "cell_complex.hpp"
#include "filtration.hpp"
    
#include <pybind11/pybind11.h>
namespace py = pybind11;
    
std::tuple<std::vector<std::vector<int> >, std::vector<int>, std::vector<int> > 
    calc_boundary_mat(Filtration &filt, CellComplex &comp) {
    
    std::vector<std::vector<int> > columns(comp.ncells);
    std::vector<int> cell_to_col(comp.ncells);    
    std::vector<int> col_to_cell(comp.ncells);
    
    std::vector<int> cell_order = filt.get_filtration();
    for(std::size_t i = 0; i < cell_order.size(); i++) {
        
        int ci = cell_order[i];

        if(comp.regular) {
            auto facet_range = comp.get_facet_range(ci);
            for(auto it = facet_range.first; it != facet_range.second; it++) {
                columns[i].push_back(cell_to_col[*it]);
            }
        } else {
            auto facet_range = comp.get_facet_range(ci);
            auto coeff_range = comp.get_coeff_range(ci);
            for(auto itf = facet_range.first, itc = coeff_range.first; itf != facet_range.second; itf++, itc++) {
                if((*itc) % 2!= 0) {
                    columns[i].push_back(cell_to_col[*itf]);
                }
            }
        }
        
        columns[i].shrink_to_fit();
        std::sort(columns[i].begin(), columns[i].end());
        cell_to_col[ci] = i;
        col_to_cell[i] = ci;
        
    }
    
    return std::forward_as_tuple(columns, cell_to_col, col_to_cell);
    
}


std::vector<std::vector<int> > calc_boundary_mat(CellComplex &comp) {
    
    std::vector<std::vector<int> > columns(comp.ncells);
    
    for(int ci = 0; ci < comp.ncells; ci++) {
        

        if(comp.regular) {
            auto facet_range = comp.get_facet_range(ci);
            for(auto it = facet_range.first; it != facet_range.second; it++) {
                columns[ci].push_back(*it);
            }
        } else {
            auto facet_range = comp.get_facet_range(ci);
            auto coeff_range = comp.get_coeff_range(ci);
            for(auto itf = facet_range.first, itc = coeff_range.first; itf != facet_range.second; itf++, itc++) {
                if((*itc) % 2!= 0) {
                    columns[ci].push_back(*itf);
                }
            }
        }
        
        columns[ci].shrink_to_fit();
        std::sort(columns[ci].begin(), columns[ci].end());
    }
    
    return columns;
    
}


std::vector<int> add_cols_Z2(std::vector<int> &col1, std::vector<int> &col2) {
    
    std::vector<int> col_sum(col1.size() + col2.size());
    auto it = std::set_symmetric_difference (col1.begin(), col1.end(), 
                                             col2.begin(), col2.end(), 
                                             col_sum.begin());
    col_sum.resize(it-col_sum.begin());
    
    return col_sum;
    
}


std::vector<std::vector<int> > reduce_smith_normal_form(std::vector<std::vector<int> > &columns, bool birth_cycles=false) {
    
    // row to reduced column with pivot in that row
    std::unordered_map<int, int> pivot_col;
     
    std::vector<std::vector<int> > g;
    
    for(unsigned int j = 0; j < columns.size(); j++) {
        
        if(birth_cycles) {
            g.emplace_back(1, j);
        }
        
        while(columns[j].size()) {
            
            int pivot_row = columns[j].back();
            
            if(!pivot_col.count(pivot_row)) {
                break;
            }

            int l = pivot_col[pivot_row];
            
            // py::print(j, columns[j]);
            // py::print(l, columns[l]);
            
            std::vector<int> col_sum = add_cols_Z2(columns[j], columns[l]);
            columns[j].assign(col_sum.begin(), col_sum.end());
            
            // py::print(j, "+", l, col_sum);
            
            if(birth_cycles) {
                std::vector<int> g_sum = add_cols_Z2(g[j], g[l]);
                g[j].assign(g_sum.begin(), g_sum.end());
            }
       
        }

        if(columns[j].size()) {
            pivot_col[columns[j].back()] = j;
        }
                
        columns[j].shrink_to_fit();
                            
    }
        
    return g;
    
}


std::vector<std::pair<int, int> > calc_persistence(Filtration &filt, CellComplex &comp) {
    
    // Get boundary matrices for ascending and descending filtrations
    auto bound_mat = calc_boundary_mat(filt, comp);

    // Initialize columns and column maps with ascending filtration
    std::vector<std::vector<int> > columns = std::get<0>(bound_mat);
    std::vector<int> col_to_cell = std::get<2>(bound_mat);

        
    reduce_smith_normal_form(columns);
    
    // py::print(columns);    
    
    std::vector<std::pair<int, int> > pairs; 
    for(std::size_t j = 0; j < columns.size(); j++) {
        if(columns[j].empty()) {
            continue;
        }
        
        int i = columns[j].back();
        int ci = col_to_cell[i];
        int cj = col_to_cell[j];
        
        pairs.emplace_back(ci, cj);
        
    }
    
    return pairs;
    
    
}

std::vector<int> calc_betti_numbers(CellComplex &comp) {
    
    // Get boundary matrices for ascending and descending filtrations
    std::vector<std::vector<int> > columns = calc_boundary_mat(comp);
        
    reduce_smith_normal_form(columns);
    
    // py::print(columns);    
    
    std::vector<int> betti(comp.dim+1, 0); 
    for(std::size_t cj = 0; cj < columns.size(); cj++) {
        if(columns[cj].empty()) {
            betti[comp.get_dim(cj)]++;
        } else {
            betti[comp.get_dim(cj)-1]--;
        }        
    }
    
    return betti;
    
    
}


// // Note: This algorithm does not work for a discrete Morse complex
// // Maxima and minima must both be the same type of cell for this to work (e.g. standard cell complex or Morse-Smale complex)
std::tuple<std::tuple<std::vector<std::pair<int, int> >, 
    std::vector<std::pair<int, int> >, 
    std::vector<std::pair<int, int> > >,
    std::unordered_map<int, std::vector<int> > >
    calc_extended_persistence(Filtration &filt_asc, Filtration &filt_desc, CellComplex &comp, bool ext_cycles, int dim=-1) {

    // Get boundary matrices for ascending and descending filtrations
    auto bound_mat_asc = calc_boundary_mat(filt_asc, comp);
    auto bound_mat_desc = calc_boundary_mat(filt_desc, comp);

    // Initialize columns and column maps with ascending filtration
    std::vector<std::vector<int> > columns = std::get<0>(bound_mat_asc);
    std::vector<int> col_to_cell = std::get<2>(bound_mat_asc);

    
    // Create extended boundary matrix
    columns.resize(2*comp.ncells);
    col_to_cell.resize(2*comp.ncells);
    for(int i = 0; i < comp.ncells; i++) {
        
        // Get cell in descending filtration
        int ci = std::get<2>(bound_mat_desc)[i];
        // Get row of cell in ascending ascending boundary matrix
        int irow = std::get<1>(bound_mat_asc)[ci];
        // Add cell to permutation matrix
        columns[i+comp.ncells].push_back(irow);
        
        // Add column to extended boundary matrix at an offset
        std::vector<int> col = std::get<0>(bound_mat_desc)[i];
        for(auto cj: col) {
            columns[i+comp.ncells].push_back(cj+comp.ncells);
        }
        columns[i+comp.ncells].shrink_to_fit();
        
        col_to_cell[i+comp.ncells] = ci;
        
    }
    
    // py::print(columns);
        
    reduce_smith_normal_form(columns);
    
    // py::print(columns);    
    
    std::vector<std::pair<int, int> > ord_pairs; 
    std::vector<std::pair<int, int> > rel_pairs;
    std::vector<std::pair<int, int> > ext_pairs;
    std::unordered_map<int, std::vector<int> > cycles;
    for(std::size_t j = 0; j < columns.size(); j++) {
        if(columns[j].empty()) {
            continue;
        }
        
        int i = columns[j].back();
        int ci = col_to_cell[i];
        int cj = col_to_cell[j];
        
        if(j < (std::size_t)comp.ncells) {
            ord_pairs.emplace_back(ci, cj);
        } else if(i < comp.ncells) {
            ext_pairs.emplace_back(ci, cj);
            
            if(ext_cycles && (dim == -1 || comp.get_dim(ci) == dim)) {
                cycles[ci];
                for(auto ck: columns[j]) {
                    cycles[ci].push_back(col_to_cell[ck]);
                }
            }
            
        } else {
            rel_pairs.emplace_back(ci, cj);
        }
        
        
    }
    
    return std::forward_as_tuple(std::forward_as_tuple(ord_pairs, rel_pairs, ext_pairs), cycles);
    
}


std::tuple<std::vector<std::pair<int, int> >, 
    std::vector<std::pair<int, int> >, 
    std::vector<std::pair<int, int> > >
    calc_extended_persistence(Filtration &filt_asc, Filtration &filt_desc, CellComplex &comp) {

    auto result = calc_extended_persistence(filt_asc, filt_desc, comp, false);
    
    return std::get<0>(result);
    
}


std::unordered_map<int, std::vector<int> > calc_birth_cycles(Filtration &filt, CellComplex &comp, int dim=-1) {
    
    auto bound_mat = calc_boundary_mat(filt, comp);
    
    std::vector<std::vector<int> > columns = std::get<0>(bound_mat);
    std::vector<int> col_to_cell = std::get<2>(bound_mat);
    
    std::vector<std::vector<int> > g = reduce_smith_normal_form(columns, true);
    
    // py::print(g);
    
    std::unordered_map<int, std::vector<int> > cycles;
    
    for(std::size_t j = 0; j < columns.size(); j++) {
        if(columns[j].size()) {
            continue;
        }
        
        int cj = col_to_cell[j];
        
        if(dim != -1 && comp.get_dim(cj) != dim) {
            continue;
        }
        
        cycles[cj];
        for(auto gi: g[j]) {
            cycles[cj].push_back(col_to_cell[gi]);
        }
    }
    
    return cycles;
}

std::unordered_map<int, std::vector<int> > calc_homologous_birth_cycles(Filtration &filt, CellComplex &comp, int dim=-1) {
    
    auto bound_mat = calc_boundary_mat(filt, comp);
    
    std::vector<std::vector<int> > columns = std::get<0>(bound_mat);
    std::vector<int> col_to_cell = std::get<2>(bound_mat);
    
    reduce_smith_normal_form(columns, false);
    
    // py::print(g);
    
    std::unordered_map<int, std::vector<int> > cycles;
    
    for(std::size_t j = 0; j < columns.size(); j++) {
        if(!columns[j].size()) {
            continue;
        }
        
        int i = columns[j].back();

        int ci = col_to_cell[i];
        
        if(dim != -1 && comp.get_dim(ci) != dim) {
            continue;
        }
        
        cycles[ci];
        for(auto Ai: columns[j]) {
            cycles[ci].push_back(col_to_cell[Ai]);
        }
    }
    
    return cycles;
}


std::unordered_set<int> extract_persistence_feature(int i, int j, CellComplex &comp, Filtration &filt, int target_dim=-1, bool complement=false) {
    
    if(target_dim == -1) {
        target_dim = comp.get_dim(i);
    }
    
    std::unordered_set<int> seen;
    seen.insert(i);
    
    std::queue<int> Q;
    Q.push(i);
    
    int orderi = filt.get_order(i);
    int orderj = filt.get_order(j);
    
    while(!Q.empty()) {
        int a = Q.front();
        Q.pop();
                
        for(auto b: comp.get_cofaces(a)) {
            
            // py::print("b", b);
            
            if(filt.get_order(b) >= orderj) {
                continue;
            }
            
            for(auto c: comp.get_faces(b)) {
                
                if(filt.get_order(c) <= orderi) {
                    continue;
                }
                
                if(!seen.count(c) && c != a) {
                    Q.push(c);
                    seen.insert(c);
                }
                
            }
        }
    }
    
    
    
    if(complement) {
        
        std::unordered_set<int> comp_seen;
        seen.insert(j);
    
        std::queue<int> Q;
        Q.push(j);
        
        
        while(!Q.empty()) {
            int a = Q.front();
            Q.pop();

            for(auto b: comp.get_faces(a)) {

                // py::print("b", b);

                if(filt.get_order(b) <= orderi) {
                    continue;
                }

                for(auto c: comp.get_cofaces(b)) {

                    if(filt.get_order(c) >= orderj) {
                        continue;
                    }

                    if(!seen.count(c) && !comp_seen.count(c) && c != a) {
                        Q.push(c);
                        comp_seen.insert(c);
                    }

                }
            }
        }
        
        seen = comp_seen;
        
    }
    
    
    
    std::unordered_set<int> feature;
    for(auto s: seen) {
        if(comp.get_dim(s) == target_dim) {
            feature.insert(s);
        }
    }
    
    return feature;
    
    
// if i is vertex, then start from i and use cofaces
// if i is not vertex, then 
    
        
//     bool co = (comp.get_dim(i) != 0);
    
//     if(target_dim == -1) {
//         target_dim = co ? comp.dim : 0;
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
  
//     int orderi = filt.get_total_order(i);
//     int orderj = filt.get_total_order(j);
         
//     while(!Q.empty()) {
//         int a = Q.front();
//         Q.pop();
        
//         // py::print("a", a);
        
//         for(auto b: get_star(a, co, comp, -1)) {
            
//             // py::print("b", b);
            
//             if((!co && filt.get_total_order(b) >= orderj)
//               || (co && filt.get_total_order(b) <= orderi)) {
//                 continue;
//             }

//             for(auto c: get_star(b, !co, comp, -1)) {
//                 // py::print("c", c);
                
//                 if((!co && filt.get_total_order(c) <= orderi)
//                   || (co && filt.get_total_order(c) >= orderj)) {
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

//             for(auto b: get_star(a, !co, comp, -1)) {

//                 // py::print("b", b);

//                 for(auto c: get_star(b, co, comp, -1)) {
//                     // py::print("c", c);
                    
//                     if((!co && filt.get_total_order(c) >= orderj)
//                       || (co && filt.get_total_order(c) <= orderi)) {
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
//         if(comp.get_dim(s) == target_dim) {
//             feature.insert(s);
//         }
//     }
    
//     return feature;
    
}



    
#endif // PERSIST_HPP