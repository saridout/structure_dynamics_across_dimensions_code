#ifndef FILTRATION_HPP
#define FILTRATION_HPP
    
    
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <numeric>
    
#include "cell_complex.hpp"
    
    
class Filtration {
    

    
protected:
    
    
    // Filtration function of cells
    XVec func;
    // Digitized (integer) representation of filtration function
    XiVec digi_func;
    // Total filtration orderering of cells
    XiVec order;
    
public:
        
    // Number of cells
    const int ncells;
    // Ascending or descending filtration?
    const bool ascend;
    // Filtration dimension (-1 if not defined)
    const int filt_dim;
    
    Filtration(CellComplex &comp, RXVec func, RXiVec digi_func, RXiVec order, 
               bool ascend = true, int filt_dim = -1) : 
       func(func), digi_func(digi_func), order(order),  ncells(comp.ncells), ascend(ascend), filt_dim(filt_dim) {}
    
    bool operator()(const int &lhs, const int &rhs) {
        return this->order[lhs] < this->order[rhs];
    }
    
    // Get value of filtration function on cell alpha
    double get_func(int alpha) {
        return func(alpha);
    }
    
    // Get value of digitized filtration function on cell alpha
    int get_digi_func(int alpha) {
        return digi_func(alpha);
    }
    
    // Get the total ordering of cell alpha
    int get_order(int alpha) {
        return order(alpha);
    }
    
    std::vector<int> get_filtration() {
        std::vector<int> filtration(ncells);
        std::iota(filtration.begin(), filtration.end(), 0);
        std::sort(filtration.begin(), filtration.end(), *this);
        
        return filtration;
    }
    
};




Filtration construct_filtration(CellComplex &comp, RXVec func, bool ascend = true) {
    
    XiVec order = XiVec::Zero(comp.ncells);
    
    auto cmp = [&comp, &func, ascend] (const int &lhs, const int &rhs) {
        
        // If lhs is lower dimension than rhs
        // Check if lhs is face of rhs
        if(comp.get_dim(lhs) < comp.get_dim(rhs)) {
            std::unordered_set<int> faces = comp.get_faces(rhs);
            if(faces.count(lhs)) {
                return true;
            }
            
        // If rhs is lower dimension than rhs
        // Check if rhs is face of lhs
        } else if(comp.get_dim(lhs) > comp.get_dim(rhs)) {
            
            std::unordered_set<int> faces = comp.get_faces(lhs);
            if(faces.count(rhs)) {
                return false;
            }
        }
           
        
        // If functions are different values
        if(func(lhs) !=  func(rhs)) {
            
            if(ascend) {
                return func(lhs) < func(rhs);
            } else {
                return func(lhs) > func(rhs);
            }
            
        // If funciton values are the same, then break tie arbitrarily
        } else {
            return lhs < rhs;
        }
        
    };
    
    std::vector<int> cells(comp.ncells);
    std::iota(cells.begin(), cells.end(), 0);
    std::sort(cells.begin(), cells.end(), cmp);
    
    for(int i = 0; i < comp.ncells; i++) {
        order(cells[i]) = i;
    }
    
    return Filtration(comp, func, order, order, ascend);
    
    
}



// Construct either lower star or upper costar filtration
// Cells of dimension filt_dim induce ordering on cells of other dimensions
Filtration construct_induced_filtration(CellComplex &comp,  RXVec func, RXiVec digi_func, int filt_dim, bool ascend=true) {
    
    XVec induced_func = XVec::Zero(comp.ncells);  
    XiVec induced_digi_func = XiVec::Zero(comp.ncells);
    XiVec order = XiVec::Zero(comp.ncells);
    
    auto lstar_cmp = [&digi_func, &comp, ascend](const int &lhs, const int &rhs) {
        
        if(ascend) {
            // Sort from high to low
            return digi_func[comp.get_label(lhs)] > digi_func[comp.get_label(rhs)]; 
        } else {
            // Sort from low to high
            return digi_func[comp.get_label(lhs)] < digi_func[comp.get_label(rhs)]; 
        }
    };
    
    auto ucostar_cmp = [&digi_func, &comp, ascend](const int &lhs, const int &rhs) {
        if(ascend) {
            // Sort from low to high
            return digi_func[comp.get_label(lhs)] < digi_func[comp.get_label(rhs)]; 
        } else {
            // Sort from high to low
            return digi_func[comp.get_label(lhs)] > digi_func[comp.get_label(rhs)]; 
        }
    };
        
    
    // Calculate lexicographic value of each cell
    std::vector<std::vector<int> > lex_val(comp.ncells);
    for(int c = 0; c < comp.ncells; c++) {
                
        std::vector<int> lex_cells;
        if(comp.get_dim(c) < filt_dim) {
            std::unordered_set<int> tmp = comp.get_cofaces(c, filt_dim);
            lex_cells.insert(lex_cells.end(), tmp.begin(), tmp.end());
            
            // Sort cells
            std::sort(lex_cells.begin(), lex_cells.end(), ucostar_cmp);
        } else {
            std::unordered_set<int> tmp = comp.get_faces(c, filt_dim);
            lex_cells.insert(lex_cells.end(), tmp.begin(), tmp.end());
            
            // Sort cells
            std::sort(lex_cells.begin(), lex_cells.end(), lstar_cmp);
        }
                                 
        // Convert to integer function values
        for(auto a: lex_cells) {
            lex_val[c].push_back(digi_func[comp.get_label(a)]);
        }
        
        if(lex_val[c].size() == 0) {
        
            py::print(c, comp.get_dim(c), lex_val[c], py::arg("flush")=true);
        }
                        
        // Record function value of cell
        induced_func(c) = func(comp.get_label(lex_cells[0]));
        induced_digi_func(c) = digi_func[comp.get_label(lex_cells[0])];
    }
            
    auto lex_cmp = [&comp, &lex_val, ascend](const int &lhs, const int &rhs) {
        
        // Ascending filtraiton
        if(ascend) {
            // Compare filtration order of cells (smallest goes first)
            if(lex_val[lhs].front() != lex_val[rhs].front()) {
                return lex_val[lhs].front() < lex_val[rhs].front();
            // Compare dimensions (smallest goes first)
            } else if(comp.get_dim(lhs) != comp.get_dim(rhs)) {
                return comp.get_dim(lhs) < comp.get_dim(rhs);
            // Compare lexicographic values (smallest goes first)
            } else if(lex_val[lhs] != lex_val[rhs]) {
                return lex_val[lhs] < lex_val[rhs];
            // Finally, if cells have identical lexicographic orderings, 
            // then sort by raw cell index
            } else {
                // py::print("Breaking tie with indices", lhs, rhs);
                return lhs < rhs;
            }
        } else {
            // Compare filtraiton value of cells (largest goes first)
            if(lex_val[lhs].front() != lex_val[rhs].front()) {
                return lex_val[lhs].front() > lex_val[rhs].front();
            // Compare dimensions (smallest goes first)
            } else if(comp.get_dim(lhs) != comp.get_dim(rhs)) {
                return comp.get_dim(lhs) < comp.get_dim(rhs);
            // Compare lexicographic values (largest goes first)
            } else if(lex_val[lhs] != lex_val[rhs]) {
                return lex_val[lhs] > lex_val[rhs];
            // Finally, if cells have identical lexicographic orderings, 
            // then sort by raw cell index
            } else {
                // py::print("Breaking tie with indices", lhs, rhs);
                return lhs > rhs;
            }
        }
            
    };
        
    // Sort cells
    std::vector<int> cells(comp.ncells);
    std::iota(cells.begin(), cells.end(), 0);
    std::sort(cells.begin(), cells.end(), lex_cmp);
    
    for(int i = 0; i < comp.ncells; i++) {
        order(cells[i]) = i;
    }
        
    return Filtration(comp, induced_func, induced_digi_func, order, ascend, filt_dim);
    

}



Filtration reduce_filtration(Filtration &full_filt, CellComplex &full_comp, CellComplex &red_comp) {
    
    
    XVec func = XVec::Zero(red_comp.ncells);  
    XiVec digi_func = XiVec::Zero(red_comp.ncells);
    XiVec order = XiVec::Zero(red_comp.ncells);
    for(int c = 0; c < full_comp.ncells; c++) {
        
        if(full_comp.get_dim(c) <= red_comp.dim && full_comp.get_label(c) != -1) {
            func(full_comp.get_label(c)) = full_filt.get_func(c);
            digi_func(full_comp.get_label(c)) = full_filt.get_digi_func(c);
            order(full_comp.get_label(c)) = full_filt.get_order(c);
        }
        
    }
    
    int filt_dim = full_filt.filt_dim >= red_comp.dim ? red_comp.dim : full_filt.filt_dim;
    
    return Filtration(red_comp, func, digi_func, order, full_filt.ascend, filt_dim);
    
}




// std::vector<int> perform_watershed_transform(std::vector<double> &time, CellComplex &comp, bool ascend = true, bool co = false) {    
    
//     int fdim = co ? comp.dim : 0;
    
//     std::vector<int> subcomplex_order(time.size());
    
//     std::unordered_map<int, int> cell_to_index;
//     std::vector<bool> submerged(time.size(), false);
            
//     // Find primary cells and sort according to insertion time
//     std::vector<int> filt_cell_argsort;
//     for(int i = 0, index = 0; i < comp.ncells; i++) {
//         if(comp.get_dim(i) == fdim) {
//             filt_cell_argsort.push_back(i);
//             cell_to_index[i] = index;
//             index++;
//         }
//     }
//     std::sort(filt_cell_argsort.begin(), filt_cell_argsort.end(),
//        [&time, &cell_to_index, ascend](const int &lhs, const int &rhs) {
//            if(ascend) {
//                // <=?
//                return time[cell_to_index[lhs]] < time[cell_to_index[rhs]];
//            } else {
//                // >=?
//                return time[cell_to_index[lhs]] > time[cell_to_index[rhs]];
//            }
//        });
                
//     // Iterate through each level set
//     unsigned int ti = 0;
//     while(ti < filt_cell_argsort.size()) {
        
//         std::unordered_set<int> level;
        
//         unsigned int tj = ti;
//         while(true) {
//             int ci = filt_cell_argsort[tj];
//             double t = time[cell_to_index[ci]];
//             level.insert(ci);
            
//             if((tj+1 == filt_cell_argsort.size()) || (t != time[cell_to_index[filt_cell_argsort[tj+1]]])) {
//                 break;
//             }
            
//             tj++;
//         }
                
//         if(level.size() == 1) {
//             int ci = *(level.begin());
//             submerged[cell_to_index[ci]] = true;
//             subcomplex_order[cell_to_index[ci]] = ti;
//             ti++;
            
//             continue;
//         }
        
//         std::unordered_map<int, double> dist;
//         std::queue<int> Q;
//         for(auto a: level) {
            
//             auto rangea = co ? comp.get_facet_range(a) : comp.get_cofacet_range(a);
//             for(auto ita = rangea.first; ita != rangea.second; ita++) {
                
//                 int b = *ita;
                
//                 auto rangeb = co ? comp.get_cofacet_range(b) : comp.get_facet_range(b);
//                 for(auto itb = rangeb.first; itb != rangeb.second; itb++) {
                    
//                     int c = *itb;
                    
//                     if(submerged[cell_to_index[c]]) {
                        
//                         double new_dist = 1.0;
                        
//                         if(!dist.count(a) || new_dist < dist[a]) {
//                             dist[a] = new_dist;
//                             Q.push(a);
//                         }
                        
//                     }

//                 }
                
//             }
            
//         }
        
//         while(!Q.empty()) {
//             int a = Q.front();
//             Q.pop();
            
//             // This differs from the python version;
//             double current_dist = dist[a];
            
//             if(!level.count(a)) {
//                 continue;
//             }
            
            
//             level.erase(a);
//             submerged[cell_to_index[a]] = true;
//             subcomplex_order[cell_to_index[a]] = ti;
//             ti++;
            
//             auto rangea = co ? comp.get_facet_range(a) : comp.get_cofacet_range(a);
//             for(auto ita = rangea.first; ita != rangea.second; ita++) {
                
//                 int b = *ita;
                
//                 auto rangeb = co ? comp.get_cofacet_range(b) : comp.get_facet_range(b);
//                 for(auto itb = rangeb.first; itb != rangeb.second; itb++) {
                    
//                     int c = *itb;
                    
//                     if(level.count(c)) {
                        
//                         double new_dist = current_dist + 1.0;
                        
//                         if(!dist.count(c) || new_dist < dist[c]) {
//                             dist[c] = new_dist;
//                             Q.push(c);
//                         }
//                     }
                    
//                 }
                
//             }
            
            
            
//         }
        
        
//         while(!level.empty()) {
//             int s = *(level.begin());
//             level.erase(s);
            
//             Q.push(s);
            
//             while(!Q.empty()) {
//                 int a = Q.front();
//                 Q.pop();

//                 // This differs from the python version;
//                 double current_dist = dist[a];

//                 level.erase(a);
//                 submerged[cell_to_index[a]] = true;
//                 subcomplex_order[cell_to_index[a]] = ti;
//                 ti++;



//                 auto rangea = co ? comp.get_facet_range(a) : comp.get_cofacet_range(a);
//                 for(auto ita = rangea.first; ita != rangea.second; ita++) {

//                     int b = *ita;

//                     std::vector<int>::iterator beginb;
//                     std::vector<int>::iterator endb;

//                     auto rangeb = co ? comp.get_cofacet_range(b) : comp.get_facet_range(b);
//                     for(auto itb = rangeb.first; itb != rangeb.second; itb++) {

//                         int c = *itb;

//                         if(level.count(c)) {
//                             double new_dist = current_dist + 1.0;

//                             if(!dist.count(c) || new_dist < dist[c]) {
//                                 dist[c] = new_dist;
//                                 Q.push(c);
//                             }
//                         }

//                     }
//                 }
//             }
            
//         }
        
//     }    
    
    
//     return subcomplex_order;
    
// }



// StarFiltration reduce_filtration(StarFiltration &full_filt, CellComplex &full_comp, CellComplex &red_comp) {
    
//     // This other method mixes subcomplexes up because it relabels them
//     // Edges are still added in correct order
    
// //     // Reduce filtration to only cover specified complex
// //     int red_fdim = full_filt.co ? red_comp.dim : 0;
    
// //     // Get map of filtration cells to index in reduced filtration
// //     std::unordered_map<int, int> filt_cells_to_index;
// //     std::vector<int> filt_cells;
// //     for(int i = 0; i < full_comp.ncells; i++) {
// //         // Full complex must have label -1 for all cells not to be included in reduced complex below the reduced complex dimension
// //         if(full_comp.get_dim(i) == red_fdim && full_comp.get_label(i) != -1) {
// //             filt_cells_to_index[i] = filt_cells_to_index.size();
// //             filt_cells.push_back(i);
// //         }
// //     }
    
// //     auto cmp = [&full_filt] (const int &lhs, const int &rhs) { 
// //                   return full_filt.get_total_order(lhs) < full_filt.get_total_order(rhs);
// //               };
    
    
// //     // Sort filtration cells
// //     std::sort(filt_cells.begin(), filt_cells.end(), cmp);
    
// //     // Get new subcomplex order and time
// //     std::vector<double> time(filt_cells.size());
// //     std::vector<int> subcomplex_order(filt_cells.size());
// //     for(unsigned int i = 0; i < filt_cells.size(); i++) {
// //         time[filt_cells_to_index[filt_cells[i]]] = full_filt.get_time(filt_cells[i]);
// //         subcomplex_order[filt_cells_to_index[filt_cells[i]]] = i;
         
// //     }
    
    
//     // This second method also mixes subcomplexes, but can just pretend that the corners represent the true subcomplexes
//     // Hence the fact that the raw original subcomplex order numbers are used
//     // However, each vertex is still mapped to a specific edge
    
//     // Reduce filtration to only cover specified complex
//     int red_fdim = full_filt.co ? red_comp.dim : 0;
    
//     // Get subcomplex order and time    
//     std::vector<double> time;
//     std::vector<int> subcomplex_order;
//     for(int i = 0; i < full_comp.ncells; i++) {
//         // Full complex must have label -1 for all cells not to be included in reduced complex below the reduced complex dimension
//         if(full_comp.get_dim(i) == red_fdim && full_comp.get_label(i) != -1) {
//             time.push_back(full_filt.get_time(i));
//             subcomplex_order.push_back(full_filt.get_subcomplex_order(i));
//         }
//     }
    
//     StarFiltration red_filt(time, subcomplex_order, red_comp, full_filt.ascend, full_filt.co);
    
    
//     std::vector<int> red_cells;
//     for(int i = 0; i < full_comp.ncells; i++) {
//         if(full_comp.get_dim(i) <= red_comp.dim && full_comp.get_label(i) != -1) {
//             red_cells.push_back(i);
//         }
//     }
    
//     auto cmp = [&full_filt] (const int &lhs, const int &rhs) { 
//                   return full_filt.get_total_order(lhs) < full_filt.get_total_order(rhs);
//               };
    
//     std::sort(red_cells.begin(), red_cells.end(), cmp);
    
    
//     for(int i = 0; i < red_comp.ncells; i++) {
//         int ci = red_cells[i];
        
//         red_filt.set_total_order(full_comp.get_label(ci), i);
        
//         std::unordered_set<int> star = get_star(i, !red_filt.co, red_comp, red_filt.fdim);
//         int filt_cell = *std::min_element(star.begin(), star.end(), cmp);
//         red_filt.add_to_subcomplex(i, filt_cell);
        
//     }
                         
//     return red_filt;
    
// }


    
#endif // FILTRATION_HPP