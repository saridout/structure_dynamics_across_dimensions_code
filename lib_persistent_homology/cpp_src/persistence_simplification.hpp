#ifndef SIMP_HPP
#define SIMP_HPP

#include "eigen_macros.hpp"
#include "cell_complex.hpp"
#include "filtration.hpp"
#include "morse_complex.hpp"

#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <numeric>
#include <utility>
#include <math.h>
#include <time.h>
   
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
    
namespace py = pybind11;
    

// Find cancellable close pair
// If strict is set to true then simply
std::pair<int, int> find_cancel_pair(int s, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, bool strict = true) {
    
    if(comp.get_dim(s) == 0) {
        return std::make_pair(-1, -1);
    }

    bool single_Vpath = false;
    int close_alpha = -1;
    int close_alpha_time = 0;
    std::vector<std::tuple<int, int, int> > morse_boundary = find_morse_boundary(s, V, comp, false, comp.oriented);
    for(auto trip: morse_boundary) {
        int c, k;
        std::tie(c, k, std::ignore) = trip;
        
 
        // If alpha is latest boundary so far, then record time and cell        
        if(k % 2 != 0 && (close_alpha == -1 || filt.get_order(c) > close_alpha_time)) {
            
            close_alpha = c;
            close_alpha_time = filt.get_order(c);
            
            // If there is only one V-path, then a cancellable close pair may exist
            single_Vpath = (k == 1);
        }
    }

    if(close_alpha == -1 || !single_Vpath) {
        return std::make_pair(-1, -1);
    }

    int close_beta = -1;
    int close_beta_time = 0;
    morse_boundary = find_morse_boundary(close_alpha, coV, comp, true, comp.oriented);
    for(auto trip: morse_boundary) {
        int c, k;
        std::tie(c, k, std::ignore) = trip;
        
        // If beta is the earliest boundary so far, then record time and cell
        if(k % 2 != 0 && (close_beta == -1 || filt.get_order(c) < close_beta_time)) {
            close_beta = c;
            close_beta_time = filt.get_order(c);
                        
        }
        
        
        
    }
        
    if(s == close_beta || !strict) {
        return std::make_pair(close_alpha, s);   
    } else {
        return std::make_pair(-1, -1);
    }
    
    
}



void cancel_close_pair(std::pair<int, int> &pair, RXiVec V, RXiVec coV, CellComplex &comp) {

    // Find V-paths from s to t (s is cell of higher dimension)
    int s = pair.second;
    int t = pair.first;

    std::vector<std::pair<int, int> > reverse_pairs;
    
    std::vector<std::tuple<int, int, int> > connections = find_connections(s, t, V, coV, comp);
    
    for(auto trip: connections) {
        int a, b;
        std::tie(a, b, std::ignore) = trip;
        reverse_pairs.emplace_back(a, b);
    }
    
    for(auto pair: reverse_pairs) {
        int a = pair.first;
        int b = pair.second;

        V(b) = a;
        coV(a) = b;
        V(a) = -1;
        coV(b) = -1;
    }
    
    
}



std::tuple<double, std::pair<int, int>> find_join_pair(std::vector<int> &cells, RXiVec V, RXiVec coV, 
                           Filtration &filt, CellComplex &comp, std::size_t ntarget_cells=1, bool verbose=false) {
    
    
    XiVec V_tmp = V;
    XiVec coV_tmp = coV;

    
    auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
        return lhs > rhs;
    };
    
    double threshold = 0.0;
    // This pair indicates cells start in the same basin
    std::pair<int, int> threshold_pair(-1, -1);
    
    for(int n = 1; ; n++) {
        std::vector<int> unpaired_crit_cells;
        for(int s = 0; s < V_tmp.size(); s++) {
            if(V_tmp(s) == s) {
                unpaired_crit_cells.push_back(s);
            }
        }
        
        
        std::priority_queue<std::pair<double, std::pair<int, int> >, 
        std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
        
        
        if(verbose) {
            py::print("Pass:", n, py::arg("flush")=true);
            py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
        }
        
        
        // Pass through all unpaired critical cells and find cancellable pairs
        for(auto s: unpaired_crit_cells) {
        
                        
            auto cpair = find_cancel_pair(s, V_tmp, coV_tmp, filt, comp);
                        
            if(cpair.first != -1) {
                                
                cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
            }   
        }
        
        if(verbose) {
            py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
        }
        
            
        // First check how many basins the vertices are divided into
        std::unordered_set<int> morse_cells;
        for(auto s: cells) {

            // Check if vertex is critical
            if(V_tmp(s) == s) {
                morse_cells.insert(s);
                continue;
            }

            // If not, then start from adjacent edge and flow to critical vertex
            std::vector<std::tuple<int, int, int> > traversal = traverse_flow(V_tmp(s), V_tmp, comp, false, false);

            for(auto trip: traversal) {
                int b, c;
                std::tie(std::ignore, b, c) = trip;
                if(b == c) {
                    morse_cells.insert(c);
                }
            }
        }

        // If cells have overlapping sets of corresponding morse cells
        if(morse_cells.size() <= ntarget_cells) {
            
//             py::print(morse_cells);
            
            return std::make_tuple(threshold, threshold_pair);
        }


        if(cancel_pairs.empty()) {
            break;
        }

        auto top = cancel_pairs.top();
        cancel_pairs.pop();
        auto cpair = top.second;

        // Record latest pair
        threshold_pair = cpair;

        // Record maximum necessary simplification threshold to reach this pair
        if(top.first > threshold) {
            threshold = top.first;
        }

//         py::print(n, top.first, top.second);
        
        cancel_close_pair(cpair, V_tmp, coV_tmp, comp);
               
        
        
    }
    
    // This pair indicates simplification tangles, so never reach relevant pair
    return std::make_tuple(-2.0 , std::make_pair(-2, -2));
    
}

void simplify_morse_complex(std::pair<int, int> pair, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, int target_dim=-1, bool cancel_target_pair=false, bool verbose=false) {
    
    auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
        return lhs > rhs;
    };
    

    for(int n = 1; ; n++) {
        std::vector<int> unpaired_crit_cells;
        for(int s = 0; s < V.size(); s++) {
            if(V(s) == s) {
                unpaired_crit_cells.push_back(s);
            }
        }
        
        
        std::priority_queue<std::pair<double, std::pair<int, int> >, 
        std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
        
        
        if(verbose) {
            py::print("Pass:", n, py::arg("flush")=true);
            py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
        }
        
        
        // Pass through all unpaired critical cells and find cancellable pairs
        for(auto s: unpaired_crit_cells) {
        
                        
            auto cpair = find_cancel_pair(s, V, coV, filt, comp);
                        
            if(cpair.first != -1) {
                                
                cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
            }   
        }
        
        if(verbose) {
            py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
        }
    
        bool cancelled = false;
        while(!cancel_pairs.empty()) {
            auto top = cancel_pairs.top();
            cancel_pairs.pop();

            auto cpair = top.second;
            
//             py::print(n, top.first, top.second);

            // If not cancelling target pair, then end
            if(!cancel_target_pair && cpair == pair) {
                return;
            }
            
            
            // If canceling only features of specific dimension, then skip other features
            if(target_dim != -1 && comp.get_dim(cpair.first) != target_dim) {
                continue;
            }
            
            

            cancel_close_pair(cpair, V, coV, comp);
            
            // If cancelling target pair, then end
            if(cancel_target_pair && cpair == pair) {
                return;
            }
            
            cancelled = true;
            break;
        }
        
        if(!cancelled) {
            break;
        }
        
    }
    
    
}


void simplify_morse_complex(double threshold, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, 
                            int target_dim=-1, bool parallel=false, bool verbose=false) {
    
    auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
        return lhs > rhs;
    };
    
    // Find all critical cells
    std::unordered_set<int> crit_cells;
    for(int s = 0; s < V.size(); s++) {
        if(V(s) == s) {
            crit_cells.insert(s);
        }
    }
    
    std::unordered_set<int> unpaired_crit_cells;
    
    std::priority_queue<std::pair<double, std::pair<int, int> >, 
        std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);

    for(int n = 1; ; n++) {
        
        if(!parallel || n == 1) {
            // Clear priority queue of cancellable pairs
            while(!cancel_pairs.empty()) {
                cancel_pairs.pop();
            }  
            
            unpaired_crit_cells = crit_cells;
            
        }
        
        if(verbose) {
            py::print("Pass:", n, py::arg("flush")=true);
            py::print("Critical Cells:", crit_cells.size(), py::arg("flush")=true);
            py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
        }
        
        
        // Pass through all unpaired critical cells and find cancellable pairs
        std::vector<int> remove;
        for(auto s: unpaired_crit_cells) {
                        
            auto cpair = find_cancel_pair(s, V, coV, filt, comp);
                        
            if(cpair.first != -1) {
                                
                cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
                remove.push_back(cpair.first);
                remove.push_back(cpair.second);
                
            }   
        }
        
        for(auto s: remove) {
            unpaired_crit_cells.erase(s);
        }
        remove.clear();
        
        if(verbose) {
            py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
        }
    
        bool cancelled = false;
        while(!cancel_pairs.empty()) {
            auto top = cancel_pairs.top();
            cancel_pairs.pop();

            auto t = top.first;
            auto cpair = top.second;

            // If canceling only features of specific dimension, then skip other features
            if(target_dim != -1 && comp.get_dim(cpair.first) != target_dim) {
                continue;
            }
           
            if(t > threshold) {
                break;
            }

            cancel_close_pair(cpair, V, coV, comp);
            
            remove.push_back(cpair.first);
            remove.push_back(cpair.second);
            
            cancelled = true;
            
            if(!parallel) {
                break;
            }
        }
        
        
        for(auto s: remove) {
            crit_cells.erase(s);
        }
        
        if(!cancelled) {
            break;
        }
        
    }
    
    
}



std::tuple<std::vector<std::pair<int, int> >, std::vector<double> > 
    find_cancel_order(RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, 
                            int target_dim=-1, bool parallel=false, bool verbose=false) {
    
    
    std::vector<std::pair<int, int> > pairs;
    std::vector<double> thresholds;
    
    XiVec V_tmp = V;
    XiVec coV_tmp = coV;

    auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
        return lhs > rhs;
    };
    
    // Find all critical cells
    std::unordered_set<int> crit_cells;
    for(int s = 0; s < V_tmp.size(); s++) {
        if(V_tmp(s) == s) {
            crit_cells.insert(s);
        }
    }
    
    std::unordered_set<int> unpaired_crit_cells;
    
    std::priority_queue<std::pair<double, std::pair<int, int> >, 
        std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);

    for(int n = 1; ; n++) {
        
        if(!parallel || n == 1) {
            // Clear priority queue of cancellable pairs
            while(!cancel_pairs.empty()) {
                cancel_pairs.pop();
            }  
            
            unpaired_crit_cells = crit_cells;
            
        }
        
        if(verbose) {
            py::print("Pass:", n, py::arg("flush")=true);
            py::print("Critical Cells:", crit_cells.size(), py::arg("flush")=true);
            py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
        }
        
        
        // Pass through all unpaired critical cells and find cancellable pairs
        std::vector<int> remove;
        for(auto s: unpaired_crit_cells) {
                        
            auto cpair = find_cancel_pair(s, V_tmp, coV_tmp, filt, comp);
                        
            if(cpair.first != -1) {
                                
                cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
                remove.push_back(cpair.first);
                remove.push_back(cpair.second);
                
            }   
        }
        
        for(auto s: remove) {
            unpaired_crit_cells.erase(s);
        }
        remove.clear();
        
        if(verbose) {
            py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
        }
    
        bool cancelled = false;
        while(!cancel_pairs.empty()) {
            auto top = cancel_pairs.top();
            cancel_pairs.pop();

            auto t = top.first;
            auto cpair = top.second;

            // If canceling only features of specific dimension, then skip other features
            if(target_dim != -1 && comp.get_dim(cpair.first) != target_dim) {
                continue;
            }

            cancel_close_pair(cpair, V_tmp, coV_tmp, comp);
            
            pairs.push_back(cpair);
            thresholds.push_back(t);
            
            remove.push_back(cpair.first);
            remove.push_back(cpair.second);
            
            cancelled = true;
            
            if(!parallel) {
                break;
            }
        }
        
        
        for(auto s: remove) {
            crit_cells.erase(s);
        }
        
        if(!cancelled) {
            break;
        }
        
    }
    
    return std::make_tuple(pairs, thresholds);
    
    
}

// void simplify_morse_complex(double threshold, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, 
//                             int target_dim=-1, bool parallel=false, bool verbose=false) {
    
//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    

//     for(int n = 1; ; n++) {
//         std::vector<int> unpaired_crit_cells;
//         for(int s = 0; s < V.size(); s++) {
//             if(V(s) == s) {
//                 unpaired_crit_cells.push_back(s);
//             }
//         }
        
        
//         std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
        
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
//         }
        
        
//         // Pass through all unpaired critical cells and find cancellable pairs
//         for(auto s: unpaired_crit_cells) {
        
                        
//             auto cpair = find_cancel_pair(s, V, coV, filt, comp);
                        
//             if(cpair.first != -1) {
                                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//             }   
//         }
        
//         if(verbose) {
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
    
//         bool cancelled = false;
//         while(!cancel_pairs.empty()) {
//             auto top = cancel_pairs.top();
//             cancel_pairs.pop();

//             auto t = top.first;
//             auto cpair = top.second;

//             // If canceling only features of specific dimension, then skip other features
//             if(target_dim != -1 && comp.get_dim(cpair.first) != target_dim) {
//                 continue;
//             }
           
//             if(t > threshold) {
//                 break;
//             }

//             cancel_close_pair(cpair, V, coV, comp);
            
//             cancelled = true;
            
//             if(!parallel) {
//                 break;
//             }
//         }
        
//         if(!cancelled) {
//             break;
//         }
        
//     }
    
    
// }


// std::tuple<double, std::pair<int, int>> find_join_feature(std::vector<int> &cells, RXiVec V, RXiVec coV, 
//                            Filtration &filt, CellComplex &comp, int N, bool verbose) {
    
    
//     XiVec V_tmp = V;
//     XiVec coV_tmp = coV;

//     std::vector<int> unpaired_crit_cells;
//     for(int s = 0; s < V.size(); s++) {
//         if(V(s) == s) {
//             unpaired_crit_cells.push_back(s);
//         }
//     }
    

//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
//     double threshold = 0.0;
//     std::pair<int, int> threshold_pair(-1, -1);
    
//     for(int n = 1; ; n++) {
                
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
                
//         // First check how many basins the vertices are divided into
//         std::unordered_set<int> morse_cells;
//         for(auto s: cells) {
            
//             py::print("cell", s, py::arg("flush")=true);
            
//             // Check if vertex is critical
//             if(V(s) == s) {
//                 morse_cells.insert(s);
//                 py::print("morse_cell", s, py::arg("flush")=true);
//                 continue;
//             }
                    
//             // If not, then start from adjacent edge and flow to critical vertex
//             std::vector<std::tuple<int, int, int> > traversal = traverse_flow(V(s), V, comp, false, false);
            
//             // py::print("V_path", traversal);
            
//             for(auto trip: traversal) {
//                 int b, c;
//                 std::tie(std::ignore, b, c) = trip;
//                 if(b == c) {
//                     py::print("morse_cell", c, py::arg("flush")=true);
//                     morse_cells.insert(c);
//                 }
//             }
//         }
        
//         // If cells have overlapping sets of corresponding morse cells
//         if(morse_cells.size() == 1) {
//             return std::make_tuple(threshold, threshold_pair);
//         }

                
//         // Otherwise pass through all unpaired critical cells and find cancellable pairs
//         std::vector<int> remove;
//         for(auto s: unpaired_crit_cells) {
            
//             if(n > 150) {
            
//                 py::print("unpaired:", s, py::arg("flush")=true);
//             }
                        
//             auto cpair = find_cancel_pair(s, V, coV, filt, comp);
            
//             if(n > 150) {
//                 py::print(cpair, py::arg("flush")=true);
//             }
            
            
//             if(cpair.first != -1) {
                
//                 if(n > 150) {
//                     py::print("hi", py::arg("flush")=true);
//                 }
            
//                  if(cpair == std::make_pair(1236, 2673)) {
//                     return std::make_tuple(-1.0 , threshold_pair);
//                  }
                
                
//                 py::print("new cancel pair", filt.get_func(cpair.second)-filt.get_func(cpair.first), cpair, py::arg("flush")=true);
                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//                 remove.push_back(cpair.first);
//                 remove.push_back(cpair.second);
                
//             }   
//         }
                
//         for(auto s: remove) {
//             unpaired_crit_cells.erase(std::find(unpaired_crit_cells.begin(), unpaired_crit_cells.end(), s));
//         }        
                
//         // Cancel critical pair with lowest persistence
        
//         if(cancel_pairs.empty()) {
//             break;
//         }
        
//         auto top = cancel_pairs.top();
//         cancel_pairs.pop();
//         threshold_pair = top.second;
                
//         if(top.first > threshold) {
//             threshold = top.first;
//         }
        
//         py::print("cancel pair", top.first, top.second, py::arg("flush")=true);
                
//         cancel_close_pair(threshold_pair, V, coV, comp);
        
//         if(n==N) {
//             return std::make_tuple(-1.0 , threshold_pair);
//         }
                
//     } 
    
//     return std::make_tuple(-1.0 , threshold_pair);
    
// }


// std::tuple<double, std::pair<int, int>> find_join_feature(std::vector<int> &cells1, std::vector<int> &cells2, 
//                                                           RXiVec V, RXiVec coV, 
//                            Filtration &filt, CellComplex &comp, bool verbose) {
    
    
//     XiVec V_tmp = V;
//     XiVec coV_tmp = coV;

//     std::vector<int> unpaired_crit_cells;
//     for(int s = 0; s < V.size(); s++) {
//         if(V(s) == s) {
//             unpaired_crit_cells.push_back(s);
//         }
//     }
    

//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
//     double threshold = 0.0;
//     std::pair<int, int> threshold_pair(-1, -1);
    
//     for(int n = 1; ; n++) {
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
        
                
//         // First check how many basins the vertices are divided into
//         std::set<int> morse_cells1;
//         for(auto s: cells1) {
            
//             // py::print("cell", s);
            
//             // Check if vertex is critical
//             if(V_tmp(s) == s) {
//                 morse_cells1.insert(s);
//                 continue;
//             }
                    
//             // If not, then start from adjacent edge and flow to critical vertex
//             std::vector<std::tuple<int, int, int> > traversal = traverse_flow(V_tmp(s), V_tmp, comp, false, false);
            
//             // py::print("V_path", traversal);
            
//             for(auto trip: traversal) {
//                 int b, c;
//                 std::tie(std::ignore, b, c) = trip;
//                 if(b == c) {
//                     // py::print("morse_cell", c);
//                     morse_cells1.insert(c);
//                 }
//             }
//         }
        
//         std::set<int> morse_cells2;
//         for(auto s: cells2) {
            
//             // py::print("cell", s);
            
//             // Check if vertex is critical
//             if(V_tmp(s) == s) {
//                 morse_cells2.insert(s);
//                 continue;
//             }
                    
//             // If not, then start from adjacent edge and flow to critical vertex
//             std::vector<std::tuple<int, int, int> > traversal = traverse_flow(V_tmp(s), V_tmp, comp, false, false);
            
//             // py::print("V_path", traversal);
            
//             for(auto trip: traversal) {
//                 int b, c;
//                 std::tie(std::ignore, b, c) = trip;
//                 if(b == c) {
//                     // py::print("morse_cell", c);
//                     morse_cells2.insert(c);
//                 }
//             }
//         }
        
        
        
//         // If cells have overlapping sets of corresponding morse cells
//         if(std::includes(morse_cells1.begin(), morse_cells1.end(), morse_cells2.begin(), morse_cells2.end())
//           || std::includes(morse_cells2.begin(), morse_cells2.end(), morse_cells1.begin(), morse_cells1.end())) {
//             return std::make_tuple(threshold, threshold_pair);
//         }

                
//         // Otherwise pass through all unpaired critical cells and find cancellable pairs
//         std::vector<int> remove;
//         for(auto s: unpaired_crit_cells) {
                        
//             auto cpair = find_cancel_pair(s, V_tmp, coV_tmp, filt, comp);
//             if(cpair.first != -1) {
            
//                 // py::print(filt.get_func(cpair.second)-filt.get_func(cpair.first), cpair);
                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//                 remove.push_back(cpair.first);
//                 remove.push_back(cpair.second);
                
//             }   
//         }
                
//         for(auto s: remove) {
//             unpaired_crit_cells.erase(std::find(unpaired_crit_cells.begin(), unpaired_crit_cells.end(), s));
//         }        
                
//         // Cancel critical pair with lowest persistence
        
//         if(cancel_pairs.empty()) {
//             break;
//         }
        
//         auto top = cancel_pairs.top();
//         cancel_pairs.pop();
//         threshold_pair = top.second;
                
//         if(top.first > threshold) {
//             threshold = top.first;
//         }
        
                
//         cancel_close_pair(threshold_pair, V_tmp, coV_tmp, comp);
                
//     } 
    
//     return std::make_tuple(-1.0 , threshold_pair);
    
// }


// double find_join_threshold(std::vector<int> &verts, RXiVec V, RXiVec coV, 
//                            Filtration &filt, CellComplex &comp, bool verbose) {
    
    
//     XiVec V_tmp = V;
//     XiVec coV_tmp = coV;

//     std::vector<int> unpaired_crit_cells;
//     for(int s = 0; s < V_tmp.size(); s++) {
//         if(V_tmp(s) == s) {
//             unpaired_crit_cells.push_back(s);
//         }
//     }
    

//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
//     double threshold = 0.0;
    
//     for(int n = 1; ; n++) {
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
        
                
//         // First check how many basins the vertices are divided into
//         std::unordered_set<int> basins;
//         unsigned int total_morse_cells = 0;
//         for(auto s: verts) {
            
//             // Check if vertex is critical
//             if(V_tmp(s) == s) {
//                 basins.insert(s);
//                 total_morse_cells++;
//                 continue;
//             }
                    
//             // If not, then start from adjacent edge and flow to critical vertex
//             std::vector<std::tuple<int, int, int> > traversal = traverse_flow(V_tmp(s), V_tmp, comp, false, false);
//             for(auto trip: traversal) {
//                 int b, c;
//                 std::tie(std::ignore, b, c) = trip;
//                 if(b == c) {
//                     basins.insert(c);
//                     total_morse_cells++;
//                 }
//             }
//         }
        
        
//         if(basins.size() < total_morse_cells) {
//             return threshold;
//         }

                
//         // Otherwise pass through all unpaired critical cells and find cancellable pairs
//         std::vector<int> remove;
//         for(auto s: unpaired_crit_cells) {
                        
//             auto cpair = find_cancel_pair(s, V_tmp, coV_tmp, filt, comp);
            
//             if(cpair.first != -1) {
            
//                 // py::print(filt.get_func(cpair.second)-filt.get_func(cpair.first), cpair);
                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//                 remove.push_back(cpair.first);
//                 remove.push_back(cpair.second);
                
//             }   
//         }
                
//         for(auto s: remove) {
//             unpaired_crit_cells.erase(std::find(unpaired_crit_cells.begin(), unpaired_crit_cells.end(), s));
//         }        
                
//         // Cancel critical pair with lowest persistence
        
//         if(cancel_pairs.empty()) {
//             break;
//         }
        
//         auto top = cancel_pairs.top();
//         cancel_pairs.pop();
//         auto cpair = top.second;
                
//         if(top.first > threshold) {
//             threshold = top.first;
//         }
        
                
//         cancel_close_pair(cpair, V_tmp, coV_tmp, comp);
                
//     } 
    
//     return -1.0;
    
// }




// double find_cancel_threshold(std::pair<int, int> pair, RXiVec V, RXiVec coV, 
//                            Filtration &filt, CellComplex &comp, bool verbose) {
    
    
//     XiVec V_tmp = V;
//     XiVec coV_tmp = coV;

//     std::vector<int> unpaired_crit_cells;
//     for(int s = 0; s < V.size(); s++) {
//         if(V(s) == s) {
//             unpaired_crit_cells.push_back(s);
//         }
//     }
    

//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
//     double threshold = 0.0;
    
//     for(int n = 1; ; n++) {
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
        
                
//         // First check if pair is already a cancellable pair
//         std::pair<int, int> cpair = find_cancel_pair(pair.second, V_tmp, coV_tmp, filt, comp);
//         if(cpair == pair) {
            
//             double p = std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first));
            
//             return p > threshold ? p : threshold;
//         }
                
//         // Otherwise pass through all unpaired critical cells and find cancellable pairs
//         std::vector<int> remove;
//         for(auto s: unpaired_crit_cells) {
                        
//             cpair = find_cancel_pair(s, V_tmp, coV_tmp, filt, comp);
//             if(cpair.first != -1) {
            
//                 // py::print(filt.get_func(cpair.second)-filt.get_func(cpair.first), cpair);
                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//                 remove.push_back(cpair.first);
//                 remove.push_back(cpair.second);
                
//             }   
//         }
                
//         for(auto s: remove) {
//             unpaired_crit_cells.erase(std::find(unpaired_crit_cells.begin(), unpaired_crit_cells.end(), s));
//         }        
                
//         // Cancel critical pair with lowest persistence
        
//         if(cancel_pairs.empty()) {
//             break;
//         }
        
//         auto top = cancel_pairs.top();
//         cancel_pairs.pop();
//         cpair = top.second;
        
//         // py::print("top", top, py::arg("flush")=true);
        
//         if(top.first > threshold) {
//             threshold = top.first;
//         }
        
                
//         cancel_close_pair(cpair, V_tmp, coV_tmp, comp);
                
//     } 
    
//     return -1.0;
    
// }



// void simplify_morse_complex(std::pair<int, int> pair, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, int target_dim=-1, bool cancel_target_pair=false, bool verbose=false) {
    
//     std::vector<int> unpaired_crit_cells;
//     for(int s = 0; s < V.size(); s++) {
//         if(V(s) == s) {
//             unpaired_crit_cells.push_back(s);
//         }
//     }
    
//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
//     double threshold = -1.0;
//     bool found_pair = false;
    
//     for(int n = 1; ; n++) {
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Critical Cells:", unpaired_crit_cells.size() + 2*cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         // Pass through all unpaired critical cells and find cancellable pairs
//         std::vector<int> remove;
//         for(auto s: unpaired_crit_cells) {
            
// //             py::print("unpaired", s, py::arg("flush")=true);
            
//             auto cpair = find_cancel_pair(s, V, coV, filt, comp);    
            
            
//             if(cpair.first != -1) {
                
//                 if(cpair == pair) {
//                     threshold = std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first));
//                     found_pair = true;
//                 }
                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//                 remove.push_back(cpair.first);
//                 remove.push_back(cpair.second);
                
//             }   
//         }
        
//         for(auto s: remove) {
            
//             unpaired_crit_cells.erase(std::find(unpaired_crit_cells.begin(), unpaired_crit_cells.end(), s));
            
//         }       
        
        
//         // Cancel critical pairs with persistence below threshold
        
//         if(verbose) {
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         int n_cancel = 0;
        
        
//         while(!cancel_pairs.empty()) {
            
            
//             auto top = cancel_pairs.top();
//             cancel_pairs.pop();
            
//             auto cthresh = top.first;
//             auto cpair = top.second;
            
//             // If found target pair and threshold is larger than threshold of target pair, then break
//             if(found_pair && cthresh > threshold) {
//                 break;
//             }
            
//             // If not cancelling target pair, then skip
//             if(found_pair && !cancel_target_pair && cpair == pair) {
//                 continue;
//             }
            
//             // If canceling only features of specific dimension, then skip other features
//             if(target_dim != -1 && comp.get_dim(cpair.first) != target_dim) {
//                 continue;
//             }
            
//             int ncrit = 0;
//             for(int s = 0; s < V.size(); s++) {
//                 if(V(s) == s) {
//                     ncrit++;
//                 }
//             }
            
//             py::print("Before cancel", cpair, ncrit);
            
//             cancel_close_pair(cpair, V, coV, comp);
            
//             ncrit = 0;
//             for(int s = 0; s < V.size(); s++) {
//                 if(V(s) == s) {
//                     ncrit++;
//                 }
//             }
                   
//             py::print("After cancel", ncrit);
            
//             n_cancel++;
            
            
//         }
        
        
//         if(verbose) {
//             py::print("Cancelled Pairs:", n_cancel, py::arg("flush")=true);
//             py::print("Remaining Critical Cells:", unpaired_crit_cells.size() + 2*cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         if(n_cancel == 0) {
//             return;
//         }
        
//     } 
        
// }


// void simplify_morse_complex(std::unordered_set<int>& preserve, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, bool finagle=false, bool verbose=false) {
    
//     // Find all critical cells and add to list of unpaired critical cells
//     std::unordered_set<int> crit_cells;
//     for(int s = 0; s < V.size(); s++) {
//         if(V(s) == s) {
//             crit_cells.insert(s);
//         }
//     }
    
//     // Comparison operator which compares (threshold, critical pair) tuples
//     // Always returns the pair with smallest persistence threshold (priority queues are backwards)
//     // Break ties arbitrarily using cell indices
//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     // Only cancel persistence pairs
//     bool strict = true;
    
//     for(int n = 1; ; n++) {
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Critical Cells:", crit_cells.size(), py::arg("flush")=true);
//         }
        
//         // Pass through all unpaired critical cells and find cancellable pairs
//         std::priority_queue<std::pair<double, std::pair<int, int> >, 
//             std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
        
//         for(auto s: crit_cells) {
            
//             auto cpair = find_cancel_pair(s, V, coV, filt, comp, strict);   
                        
//             // If either of the cells are to be preserved, then skip
//             if(cpair.first == -1 || preserve.count(cpair.first) || preserve.count(cpair.second)) {
//                 continue;
                
//             } 
            
//             // Add new cancellable pair
//             cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//         }
        
 
//         // Cancel critical pairs with persistence below threshold
        
//         if(verbose) {
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         // Cancel all pairs
//         int n_cancel = 0;
//         while(!cancel_pairs.empty()) {
            
//             auto top = cancel_pairs.top();
//             cancel_pairs.pop();
//             auto cpair = top.second;
//             cancel_close_pair(cpair, V, coV, comp);
               
            
//             crit_cells.erase(cpair.first);
//             crit_cells.erase(cpair.second);
            
//             n_cancel++;
            
            
//             // If not strict, only cancel a single pair
//             if(!strict) {
//                 break;
//             }
            
            
//         }
        
//         if(verbose) {
//             py::print("Cancelled Pairs:", n_cancel, py::arg("flush")=true);
//             py::print("Remaining Critical Cells:", crit_cells.size(), py::arg("flush")=true);
//         }
        
//         if(n_cancel == 0) {
//             // No pairs were cancelled and the algorithm strictly cancels persitence pairs, then finish
//             // Or if nothing was cancelled and was not strict
//             if(!finagle || !strict) {
//                 return;
                
//             // Otherwise cancel a single pair of cells that doesn't correspond to a persistence pair
//             } else {
                
//                 strict = false;
                
//             }
            
//         // If was cancelled, make sure next step is strict
//         } else {
//             strict = true;
//         }
        
//     } 
        
// }


// void simplify_morse_complex(double threshold, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, bool leq=true, bool verbose=false) {
    
//     std::vector<int> unpaired_crit_cells;
//     for(int s = 0; s < V.size(); s++) {
//         if(V(s) == s) {
//             unpaired_crit_cells.push_back(s);
//         }
//     }
    
//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
//     for(int n = 1; ; n++) {
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Critical Cells:", unpaired_crit_cells.size() + 2*cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         // Pass through all unpaired critical cells and find cancellable pairs
//         std::vector<int> remove;
//         for(auto s: unpaired_crit_cells) {
            
//             auto cpair = find_cancel_pair(s, V, coV, filt, comp);            
            
//             if(cpair.first != -1) {
                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//                 remove.push_back(cpair.first);
//                 remove.push_back(cpair.second);
                
//             }   
//         }
        
//         for(auto s: remove) {
            
            
//             unpaired_crit_cells.erase(std::find(unpaired_crit_cells.begin(), unpaired_crit_cells.end(), s));
            
//         }       
        
        
//         // Cancel critical pairs with persistence below threshold
        
//         if(verbose) {
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         int n_cancel = 0;
        
//         while(!cancel_pairs.empty() && ((leq &&  cancel_pairs.top().first <= threshold)
//                   || (!leq && cancel_pairs.top().first < threshold))) {
            
//             auto top = cancel_pairs.top();
//             cancel_pairs.pop();
//             auto cpair = top.second;
//             cancel_close_pair(cpair, V, coV, comp);
                              
//             n_cancel++;
            
//         }
        
//         if(verbose) {
//             py::print("Cancelled Pairs:", n_cancel, py::arg("flush")=true);
//             py::print("Remaining Critical Cells:", unpaired_crit_cells.size() + 2*cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         if(n_cancel == 0) {
//             return;
//         }
        
//     } 
        
// }





// double simplify_morse_complex(std::pair<int, int> pair, RXiVec V, RXiVec coV, 
//                            Filtration &filt, CellComplex &comp, bool verbose) {

//     std::vector<int> unpaired_crit_cells;
//     for(int s = 0; s < V.size(); s++) {
//         if(V(s) == s) {
//             unpaired_crit_cells.push_back(s);
//         }
//     }
    

//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
        
//     double threshold = 0.0;
//     int n_cancel = 0;
    
//     for(int n = 1; ; n++) {
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Unpaired Critical Cells:", unpaired_crit_cells.size(), py::arg("flush")=true);
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//         }
        
//         // Otherwise pass through all unpaired critical cells and find cancellable pairs
//         std::vector<int> remove;
//         for(auto s: unpaired_crit_cells) {
                        
//             std::pair<int, int> cpair = find_cancel_pair(s, V, coV, filt, comp);
//             if(cpair.first != -1) {
            
//                 // py::print(filt.get_func(cpair.second)-filt.get_func(cpair.first), cpair);
                
//                 cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
                
//                 remove.push_back(cpair.first);
//                 remove.push_back(cpair.second);
                
//             }   
//         }
                
//         for(auto s: remove) {
//             unpaired_crit_cells.erase(std::find(unpaired_crit_cells.begin(), unpaired_crit_cells.end(), s));
//         }        
                
//         // Cancel critical pair with lowest persistence
        
//         if(cancel_pairs.empty()) {
//             break;
//         }
        
//         auto top = cancel_pairs.top();
//         cancel_pairs.pop();
        
//         threshold = top.first;
//         auto cpair = top.second;
        
                
//         cancel_close_pair(cpair, V, coV, comp);
        
//         n_cancel++;
        

        
//         if(cpair == pair) {
//             break;
//         }
        
                
//     } 
    
    
//     if(verbose) {
//         py::print("Cancelled Pairs:", n_cancel, py::arg("flush")=true);
//         py::print("Remaining Critical Cells:", unpaired_crit_cells.size() + 2*cancel_pairs.size(), py::arg("flush")=true);
//     }
    
//     return threshold;
    
    
// }



#endif // SIMP_HPP