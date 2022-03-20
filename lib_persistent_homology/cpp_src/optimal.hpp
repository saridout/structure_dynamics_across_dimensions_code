#ifndef OPTIMAL_HPP
#define OPTIMAL_HPP
 
#include <vector>
#include <unordered_map>
    
#include <Eigen/Core>
#include <Eigen/Sparse>
    
typedef Eigen::MatrixXd XMat;
typedef Eigen::SparseMatrix<double> SMat;
typedef Eigen::Triplet<double> Trip;

#include <ilcplex/ilocplex.h>
    
#include "cell_complex.hpp"
#include "filtration.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;


void optimize_cycle(int j, std::vector<int> &x, std::vector<int> &y, std::vector<int> &a, 
                    SMat &g, SMat &A, SMat &z, std::vector<double> &w, bool verbose=false) {
        
    // Create map of rows in full boundary matrix to rows in boundary matrix of just cells in optimization problem
    std::unordered_map<int, int> full_to_red_index;
    for(std::size_t i = 0; i < x.size(); i++) {
        full_to_red_index[x[i]] = i;
    }
    
    // Initialize environment and problem model
    IloEnv env;
    IloNumVarArray vars(env);
    IloModel mod(env);
    
    // Set up equality constraint
    IloNumArray zj(env, x.size());
    for(SMat::InnerIterator it(g, j); it; ++it) {
        if(full_to_red_index.count(it.row())) {
            zj[full_to_red_index[it.row()]] = it.value();
        }
    }
    
//     SMat gb = g.col(j);
//     py::print(gb);
            
    // Range is vector of equality constraints, so the lower and upper bounds are both zj
    IloRangeArray range (env, zj, zj);
    // Add range to model
    mod.add(range);
    
    // Initialize cost functions
    IloObjective cost = IloAdd(mod, IloMinimize(env));
    
    // Construct model column by column
    
    // First add identity columns for x^+
    for(std::size_t i = 0; i < x.size(); i++) {
        // Add w[i]*x[i] to objective function
        IloNumColumn col = cost(w[i]);
        // Add column with 1.0 for x[i]
        col += range[i](1.0);
        
        std::stringstream ss;
        ss << "xp" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
        // Add x[i] to list of variables
        vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
        col.end();
    }
    
    // Add negative identity columns for x^-
    for(std::size_t i = 0; i < x.size(); i++) {
        // Add w[i]*x[i] to objective function
        IloNumColumn col = cost(w[i]);
        // Add column with -1.0 for x[i]
        col += range[i](-1.0);
        
        std::stringstream ss;
        ss << "xm" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
        
        // Add x[i] to list of variables
        vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
        col.end();
    }
    
    // Add columns from boundary matrix for y^+
    for(std::size_t i = 0; i < y.size(); i++) {
        // Add 0.0*y[i] to objective function
        IloNumColumn col = cost(0.0);
        // Add column with -A_j for y[i]
        for(SMat::InnerIterator it(A, y[i]); it; ++it) {
            if(full_to_red_index.count(it.row())) {
                col += range[full_to_red_index[it.row()]](-it.value());
            }
        }
        
        std::stringstream ss;
        ss << "yp" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
        // Add y[i] to list of variables
        vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
        col.end();
    }
    
    // Add columns from boundary matrix for y^-
    for(std::size_t i = 0; i < y.size(); i++) {
        // Add 0.0*y[i] to objective function
        IloNumColumn col = cost(0.0);
        // Add column with A_j for y[i]
        for(SMat::InnerIterator it(A, y[i]); it; ++it) {
            if(full_to_red_index.count(it.row())) {
                col += range[full_to_red_index[it.row()]](it.value());
            }
        }
        
        std::stringstream ss;
        ss << "ym" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
        // Add y[i] to list of variables
        vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
        col.end();
    }
    
    // Add columns from boundary matrix for a^+
    for(std::size_t i = 0; i < a.size(); i++) {
        // Add 0.0*a[i] to objective function
        IloNumColumn col = cost(0.0);
        // Add column with z_j for a[i]
        for(SMat::InnerIterator it(z, a[i]); it; ++it) {
            if(full_to_red_index.count(it.row())) {
                col += range[full_to_red_index[it.row()]](it.value());
            }
        }
        
        std::stringstream ss;
        ss << "ap" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
        // Add y[i] to list of variables
        vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
        col.end();
    }
    
    // Add columns from boundary matrix for a^-
    for(std::size_t i = 0; i < a.size(); i++) {
        // Add 0.0*a[i] to objective function
        IloNumColumn col = cost(0.0);
        // Add column with -z_j for a[i]
        for(SMat::InnerIterator it(z, a[i]); it; ++it) {
            if(full_to_red_index.count(it.row())) {
                col += range[full_to_red_index[it.row()]](-it.value());
            }
        }
        
        std::stringstream ss;
        ss << "am" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
        // Add y[i] to list of variables
        vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
        col.end();
    }
    
    range.end();
    
    
    // Solve model

    IloCplex cplex(mod);
    
    if(!verbose) {
        cplex.setOut(env.getNullStream());
    }
    
//     if(j == 226) {
//         cplex.exportModel("optimal_cycle.lp");
//     }

    cplex.solve();
    
    if(verbose) {
        cplex.out() << "solution status = " << cplex.getStatus() << std::endl;
        cplex.out() << std::endl;
        cplex.out() << "cost   = " << cplex.getObjValue() << std::endl;
        for (int i = 0; i < vars.getSize(); i++) {
            cplex.out() << "  x" << i << " = " << cplex.getValue(vars[i]) << std::endl;
        }
    }
    
    
    for(std::size_t i = 0; i < x.size(); i++) {
        z.insert(x[i], j) = cplex.getValue(vars[i]) - cplex.getValue(vars[x.size() + i]);
    }
    
    z.prune(0.0);
    
    
//     SMat zb = z.col(j);
//     py::print(zb);
    
    env.end();    
    
}

// Calculate cycles homologous at birth
// These cycles may not be homologous at death, but this algorithm is more efficient
// This doesn't work for dim = -1
std::unordered_map<int, std::vector<int> > calc_optimal_cycles(Filtration &filt, CellComplex &comp, std::vector<double> &weights, int dim=-1, bool verbose=false) {
        
    if(!comp.oriented) {
        py::print("Cell complex does not have oriented cells");
        return std::unordered_map<int, std::vector<int> >();
    }
    
    if((int)weights.size() != comp.ncells) {
        weights.assign(comp.ncells, 1.0);
    }
    
    
    
    SMat A(comp.ncells, comp.ncells);
    std::vector<int> cell_to_col(comp.ncells);    
    std::vector<int> col_to_cell(comp.ncells);
    
    // Construct full boundary matrix
    std::vector<Trip> trip_list;
    std::vector<int> cell_order = filt.get_filtration();    
    for(int j = 0; j < comp.ncells; j++) {
        
        int cj = cell_order[j];

        auto facet_range = comp.get_facet_range(cj);
        auto coeff_range = comp.get_coeff_range(cj);
        for(auto itf = facet_range.first, itc = coeff_range.first; itf != facet_range.second; itf++, itc++) {
            int ci = *itf;
            int c = *itc;
            if(c != 0) {
                trip_list.emplace_back(cell_to_col[ci], j, c);
            }
        }
        
        cell_to_col[cj] = j;
        col_to_cell[j] = cj;
        
    }
    A.setFromTriplets(trip_list.begin(), trip_list.end());
            
    // Birth cycle basis
    SMat g(comp.ncells, comp.ncells);
    g.setIdentity();
    
    SMat z(comp.ncells, comp.ncells);
            
    // Simplices to include in optimization
    std::vector<int> x;
    // Columns of A to include in optimization
    std::vector<int> y;
    // Columns of z to include in optimization
    std::vector<int> a;
    // Weights of simplices in x
    std::vector<double> w;
    
     // row to reduced column with pivot in that row
    std::unordered_map<int, int> pivot_col;    
    
    for(int j = 0; j < A.cols(); j++) { 
        
        if(dim == -1 || comp.get_dim(col_to_cell[j]) == dim) {
            x.push_back(j);
            w.push_back(weights[col_to_cell[j]]);
        }
        
        // Reduce column as much as possible
        while(A.col(j).nonZeros()) {
            
            SMat::ReverseInnerIterator it(A,j);
            int pivot_row = it.row();
                                     
            if(!pivot_col.count(pivot_row)) {
                break;
            }

            int l = pivot_col[pivot_row];
            
            double r = it.value() / SMat::ReverseInnerIterator(A,l).value();
            
            A.col(j) = (A.col(j) - r * A.col(l)).pruned();
            
            g.col(j) = (g.col(j) - r * g.col(l)).pruned();
            
            // py::print(j, "+", l, SMat(A.col(j)));
       
        }
        
        // If column is zeroed out (has no nonzeros, i.e. is a birth column) 
        // and the simplex related to the column is of the target dimension,
        // then calculate optimial cycle
        if(!A.col(j).nonZeros() && comp.get_dim(col_to_cell[j]) == dim) {
            
            py::print(j, comp.get_dim(col_to_cell[j]));

            
            optimize_cycle(j, x, y, a, g, A, z, w, verbose);
            a.push_back(j);
        }

        // If column is not zeroed out (is a death column),
        // then remove the relevant cycle from the cycle basis and add it to the boundary matrix
        if(A.col(j).nonZeros()) {
            pivot_col[SMat::ReverseInnerIterator(A,j).row()] = j;
            
            if(comp.get_dim(col_to_cell[j]) == dim+1) {
                y.push_back(j);
                a.erase(std::remove(a.begin(), a.end(), SMat::ReverseInnerIterator(A,j).row()), a.end());
            }
        }
                                            
    }
    
    std::unordered_map<int, std::vector<int> > cycles;
    
    for(int j = 0; j < comp.ncells; j++) {
        if(A.col(j).nonZeros()) {
            continue;
        }
        
        int cj = col_to_cell[j];
        
        if(dim != -1 && comp.get_dim(cj) != dim) {
            continue;
        }
        
        cycles[cj];
        for(SMat::InnerIterator it(z,j); it; ++it) {
            cycles[cj].push_back(col_to_cell[it.row()]);
        }
    }
    
    
    
        
    return cycles;
    
                
}





// void optimize_homologous_cycle(int j, std::vector<int> &x, std::vector<int> &yb, std::vector<int> &yd, 
//                                std::vector<int> &a, SMat &g, SMat &gd, SMat &A, SMat &z, std::vector<double> &w, bool verbose=false) {
        
//     // Create map of rows in full boundary matrix to rows in boundary matrix of just cells in optimization problem
//     std::unordered_map<int, int> full_to_red_index;
//     for(std::size_t i = 0; i < x.size(); i++) {
//         full_to_red_index[x[i]] = i;
//     }
    
//     // Initialize environment and problem model
//     IloEnv env;
//     IloNumVarArray vars(env);
//     IloModel mod(env);
    
//     // Set up equality constraint
// //     IloNumArray zj(env, 2*x.size());
//     IloNumArray zj(env, x.size());
    
//     // Birth cycle
// //     for(SMat::InnerIterator it(g, j); it; ++it) {
// //         if(full_to_red_index.count(it.row())) {  
// // //             py::print("b", full_to_red_index[it.row()], it.value());
// //             zj[full_to_red_index[it.row()]] = it.value();
// //         }
// //     }
    
//     for(SMat::InnerIterator it(gd, 0); it; ++it) {
//         if(full_to_red_index.count(it.row())) {  
// //             py::print("d", full_to_red_index[it.row()], it.value());
//             zj[full_to_red_index[it.row()]] = it.value();
//         }
//     }  
    
// //     SMat gb = g.col(j);
// //     py::print(gb);
    
// //     int sign = g.col(j).dot(gd.col(0)) < 0 ? -1 : 1;
// // //     int sign = 1;
// //     py::print(sign);
    
   
    
// //     // Death cycle
// //     for(SMat::InnerIterator it(gd, 0); it; ++it) {
// //         if(full_to_red_index.count(it.row())) {  
// // //             py::print("d", full_to_red_index[it.row()], it.value());
// //             zj[full_to_red_index[it.row()]+x.size()] = sign * it.value();
// //         }
// //     }    
            
//     // Range is vector of equality constraints, so the lower and upper bounds are both zj
//     IloRangeArray range (env, zj, zj);
//     // Add range to model
//     mod.add(range);
    
//     // Initialize cost functions
//     IloObjective cost = IloAdd(mod, IloMinimize(env));
    
//     // Construct model column by column
    
//     // First add identity columns for x^+
//     for(std::size_t i = 0; i < x.size(); i++) {
//         // Add w[i]*x[i] to objective function
//         IloNumColumn col = cost(w[i]);
//         // Add column with 1.0 for x[i]
//         col += range[i](1.0);
// //         col += range[i+x.size()](1.0);
//         // Add x[i] to list of variables
        
//         std::stringstream ss;
//         ss << "xp" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
//         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
//         col.end();
//     }
    
//     // Add negative identity columns for x^-
//     for(std::size_t i = 0; i < x.size(); i++) {
//         // Add w[i]*x[i] to objective function
//         IloNumColumn col = cost(w[i]);
//         // Add column with -1.0 for x[i]
//         col += range[i](-1.0);
// //         col += range[i+x.size()](-1.0);
//         // Add x[i] to list of variables
        
//         std::stringstream ss;
//         ss << "xm" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
//         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
//         col.end();
//     }
    
//     // Add columns from boundary matrix for yb^+
//     for(std::size_t i = 0; i < yb.size(); i++) {
//         // Add 0.0*yb[i] to objective function
//         IloNumColumn col = cost(0.0);
//         // Add column with -A_j for yb[i]
//         for(SMat::InnerIterator it(A, yb[i]); it; ++it) {
//             if(full_to_red_index.count(it.row())) {
//                 col += range[full_to_red_index[it.row()]](-it.value());
//             }
//         }
        
        
// //         SMat tmp = A.col(yb[i]);
// //         py::print("yb_", i);
// //         py::print(tmp);
        
//         std::stringstream ss;
//         ss << "ybp" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
//         // Add yb[i] to list of variables
//         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
//         col.end();
//     }
    
//     // Add columns from boundary matrix for yb^-
//     for(std::size_t i = 0; i < yb.size(); i++) {
//         // Add 0.0*yb[i] to objective function
//         IloNumColumn col = cost(0.0);
//         // Add column with A_j for yb[i]
//         for(SMat::InnerIterator it(A, yb[i]); it; ++it) {
//             if(full_to_red_index.count(it.row())) {
//                 col += range[full_to_red_index[it.row()]](it.value());
//             }
//         }
        
//         std::stringstream ss;
//         ss << "ybm" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
//         // Add yb[i] to list of variables
//         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
//         col.end();
//     }
    
    
// //     // Add columns from boundary matrix for yd^+
// //     for(std::size_t i = 0; i < yd.size(); i++) {
// //         // Add 0.0*yd[i] to objective function
// //         IloNumColumn col = cost(0.0);
// //         // Add column with -A_j for yd[i]
// //         for(SMat::InnerIterator it(A, yd[i]); it; ++it) {
// //             if(full_to_red_index.count(it.row())) {
// //                 col += range[full_to_red_index[it.row()]+x.size()](-it.value());
// //             }
// //         }
        
// //         SMat tmp = A.col(yd[i]);
// //         py::print("yd_", i);
// //         py::print(tmp);
        
// //         std::stringstream ss;
// //         ss << "ydp" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
// //         // Add yd[i] to list of variables
// //         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
// //         col.end();
// //     }
    
// //     // Add columns from boundary matrix for yd^-
// //     for(std::size_t i = 0; i < yd.size(); i++) {
// //         // Add 0.0*yd[i] to objective function
// //         IloNumColumn col = cost(0.0);
// //         // Add column with A_j for yd[i]
// //         for(SMat::InnerIterator it(A, yd[i]); it; ++it) {
// //             if(full_to_red_index.count(it.row())) {
// //                 col += range[full_to_red_index[it.row()]+x.size()](it.value());
// //             }
// //         }
        
// //         std::stringstream ss;
// //         ss << "ydm" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
// //         // Add yd[i] to list of variables
// //         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
// //         col.end();
// //     }
            
// //     // Add columns from boundary matrix for a^+
// //     for(std::size_t i = 0; i < a.size(); i++) {
// //         // Add 0.0*a[i] to objective function
// //         IloNumColumn col = cost(0.0);
// //         // Add column with g_j for a[i]      
// //         for(SMat::InnerIterator it(g, a[i]); it; ++it) {
// //             if(full_to_red_index.count(it.row())) {                
// //                 col += range[full_to_red_index[it.row()]](it.value());
// //             }
// //         }
        
        
// // //         SMat tmp = g.col(a[i]);
// // //         py::print("g_", i);
// // //         py::print(tmp);
        
// //         std::stringstream ss;
// //         ss << "ap" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
// //         // Add y[i] to list of variables
// //         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
// //         col.end();
// //     }
    
// //     // Add columns from boundary matrix for a^-
// //     for(std::size_t i = 0; i < a.size(); i++) {
// //         // Add 0.0*a[i] to objective function
// //         IloNumColumn col = cost(0.0);
// //         // Add column with -g_j for a[i]
// //         for(SMat::InnerIterator it(g, a[i]); it; ++it) {
// //             if(full_to_red_index.count(it.row())) {
// //                 col += range[full_to_red_index[it.row()]](-it.value());
// //             }
// //         }
        
// //         std::stringstream ss;
// //         ss << "am" <<  std::setw(5) << std::setfill('0') << i << std::setfill(' ');
        
// //         // Add y[i] to list of variables
// //         vars.add(IloNumVar(col, 0.0, IloInfinity, ILOFLOAT, ss.str().c_str()));
// //         col.end();
// //     }
    
//     range.end();
    
    
//     // Solve model

//     IloCplex cplex(mod);
    
//     if(!verbose) {
//         cplex.setOut(env.getNullStream());
//     }
    
// //     cplex.exportModel("optimal_cycle.lp");
    
// //     if(x.size() == 492 && yb.size() == 0 && yd.size() == 264 && a.size() == 47) {
// //         py::print("hi");
// //         cplex.exportModel("optimal_cycle.lp");
// //     }

//     cplex.solve();
    
//     if(verbose) {
        
//         py::print("num vars: ", vars.getSize());
//         py::print(x.size(), yb.size(), yd.size(), a.size());
        
//         cplex.out() << "solution status = " << cplex.getStatus() << std::endl;
//         cplex.out() << std::endl;
        
// //         if(cplex.getStatus() == IloAlgorithm::Infeasible) {
            
// //             IloNumArray preferences(env);
// //             for (IloInt i = 0; i<range.getSize(); i++) {
// //            preferences.add(1.0);  // user may wish to assign unique preferences
// //          }
            
// //             if ( cplex.refineConflict(range, preferences) ) {
// //                 IloCplex::ConflictStatusArray conflict = cplex.getConflict(range);
// //                 env.getImpl()->useDetailedDisplay(IloTrue);
// //             cplex.out() << "Conflict :" << std::endl;
// //             for (IloInt i = 0; i<range.getSize(); i++) {
// //               if ( conflict[i] == IloCplex::ConflictMember)
// //                    cplex.out() << "Proved  : " << range[i] << std::endl;;
// //               if ( conflict[i] == IloCplex::ConflictPossibleMember)
// //                    cplex.out() << "Possible: " << range[i] << std::endl;;
// //             }
// //          }
            
// //         }
        
        
        
//         cplex.out() << "cost   = " << cplex.getObjValue() << std::endl;
        

// //         for (int i = 0; i < vars.getSize(); i++) {
// //             cplex.out() << "  x" << i << " = " << cplex.getValue(vars[i]) << std::endl;
// //         }
//     }
        
    
//     for(std::size_t i = 0; i < x.size(); i++) {
//         z.insert(x[i], j) = cplex.getValue(vars[i]) - cplex.getValue(vars[x.size() + i]);
//     }
    
//     z.prune(0.0);
    
//     SMat zb = z.col(j);
    
//     if(zb.nonZeros() < gd.nonZeros()) {
    
//         py::print("birth cycle");
//         py::print(gd);

//         py::print("final cycle");
//         py::print(zb);

//         py::print("difference");
//         SMat diff = zb - gd;
//         diff.prune(0.0);
//         py::print(diff);
//     }
        
//     env.end();    
    
// }




// // Calculate optimal birth cycles that are homologous at death
// // This doesn't work for dim = -1
// std::unordered_map<int, std::vector<int> > calc_optimal_homologous_cycles(Filtration &filt, CellComplex &comp, std::vector<double> &weights, int dim=-1, bool verbose=false) {
        
//     if(!comp.oriented) {
//         py::print("Cell complex does not have oriented cells");
//         return std::unordered_map<int, std::vector<int> >();
//     }
    
//     if((int)weights.size() != comp.ncells) {
//         weights.assign(comp.ncells, 1.0);
//     }
    
//     SMat A(comp.ncells, comp.ncells);
//     std::vector<int> cell_to_col(comp.ncells);    
//     std::vector<int> col_to_cell(comp.ncells);
    
//     // Construct full boundary matrix
//     std::vector<Trip> trip_list;
//     std::vector<int> cell_order = filt.get_filtration();    
//     for(int j = 0; j < comp.ncells; j++) {
        
//         int cj = cell_order[j];

//         auto facet_range = comp.get_facet_range(cj);
//         auto coeff_range = comp.get_coeff_range(cj);
//         for(auto itf = facet_range.first, itc = coeff_range.first; itf != facet_range.second; itf++, itc++) {
//             int ci = *itf;
//             int c = *itc;
//             if(c != 0) {
//                 trip_list.emplace_back(cell_to_col[ci], j, c);
//             }
//         }
        
//         cell_to_col[cj] = j;
//         col_to_cell[j] = cj;
        
//     }
//     A.setFromTriplets(trip_list.begin(), trip_list.end());
        
//     // Birth cycle basis
//     SMat g(comp.ncells, comp.ncells);
//     g.setIdentity();
    
//     // Reduce boundary matrix and find birth cycle basis
//     // row to reduced column with pivot in that row
//     std::unordered_map<int, int> pivot_col;    
//     for(int j = 0; j < A.cols(); j++) { 
        
//         // Reduce column as much as possible
//         while(A.col(j).nonZeros()) {
            
//             SMat::ReverseInnerIterator it(A,j);
//             int pivot_row = it.row();
                                     
//             if(!pivot_col.count(pivot_row)) {
//                 break;
//             }

//             int l = pivot_col[pivot_row];
            
//             double r = it.value() / SMat::ReverseInnerIterator(A,l).value();
            
//             A.col(j) = (A.col(j) - r * A.col(l)).pruned();
            
//             g.col(j) = (g.col(j) - r * g.col(l)).pruned();
            
//             // py::print(j, "+", l, SMat(A.col(j)));
       
//         }
        
        
//          // If column cannot be reduced, then add its row to the pivot column row list
//         if(A.col(j).nonZeros()) {
//             pivot_col[SMat::ReverseInnerIterator(A,j).row()] = j;
            
  
//         }
        
//     }
    
//     // Simplices to include in optimization
//     std::vector<int> x;
//     // Weights of simplices in x
//     std::vector<double> w;
   
//     // Columns of A to include in optimization (death boundary mat)
//     std::vector<int> yd;
    
//     // Optimal cycles
//     SMat z(comp.ncells, comp.ncells);
    
//     // Iterate through each column and optimize birth cycles 
//     for(int j = 0; j < A.cols(); j++) { 
        
        
//         if(comp.get_dim(col_to_cell[j]) == dim) {
//             // Add cell to optimization problem
//             x.push_back(j);
//             w.push_back(weights[col_to_cell[j]]);
//         }
        
//         // If column is a death column
//         if(A.col(j).nonZeros() && comp.get_dim(col_to_cell[j]) == dim+1) {
            
            
//             int birth = SMat::ReverseInnerIterator(A,j).row();
//             int death = j;
                        
//              // Columns of A to include in optimization (birth boundary mat)
//             std::vector<int> yb;
//             // Columns of g to include in optimization
//             std::vector<int> a;
            
            
//             for(int k = 0; k < birth; k++) {
                
//                 // If column is a birth column
//                 if(!A.col(k).nonZeros() && comp.get_dim(col_to_cell[k]) == dim) {
//                     // Add column to basis of birth cycles
//                     a.push_back(k);
//                 // If column is death column
//                 } else if(A.col(k).nonZeros() && comp.get_dim(col_to_cell[k]) == dim+1) {
//                     // Add column to birth boundary matrix
//                     yb.push_back(k);
//                     // Remove cycle from birth cycle basis
//                     a.erase(std::remove(a.begin(), a.end(), SMat::ReverseInnerIterator(A,k).row()), a.end());
//                 }
//             }
            
            
//             // Construct death cycle
//             SMat gd(comp.ncells, 1);
    
// //             // Construct full boundary matrix
// //             std::vector<Trip> trip_list;

// //             int cj = cell_order[death];

// //             auto facet_range = comp.get_facet_range(cj);
// //             auto coeff_range = comp.get_coeff_range(cj);
// //             for(auto itf = facet_range.first, itc = coeff_range.first; itf != facet_range.second; itf++, itc++) {
// //                 int ci = *itf;
// //                 int c = *itc;
// //                 if(c != 0) {
// //                     trip_list.emplace_back(cell_to_col[ci], 0, c);
// //                 }
// //             }

// //             gd.setFromTriplets(trip_list.begin(), trip_list.end());
            
            
//             gd = A.col(death);
            
            
// //             py::print(birth, death, comp.get_dim(col_to_cell[birth]), comp.get_dim(col_to_cell[death]));
             
//             optimize_homologous_cycle(birth, x, yb, yd, a, g, gd, A, z, w, verbose);
            
//             // Add column to death boundary matrix
//             yd.push_back(j);
                          
//         }
        
        
//     }
        
    
//     std::unordered_map<int, std::vector<int> > cycles;
    
//     for(int j = 0; j < comp.ncells; j++) {
//         if(A.col(j).nonZeros()) {
//             continue;
//         }
        
//         int cj = col_to_cell[j];
        
//         if(comp.get_dim(cj) != dim) {
//             continue;
//         }
        
//         cycles[cj];
//         for(SMat::InnerIterator it(z,j); it; ++it) {
//             cycles[cj].push_back(col_to_cell[it.row()]);
//         }
                
//     }
        
//     return cycles;
    
                
// }

    
#endif // OPTIMAL_HPP