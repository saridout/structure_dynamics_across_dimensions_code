#ifndef PROTEIN_HPP
#define PROTEIN_HPP

#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <queue>
#include <tuple> 
#include <algorithm>
#include <random>

#include "eigen_macros.hpp"
#include "cell_complex.hpp"
#include "filtration.hpp"
#include "search.hpp"
#include "persistence_simplification.hpp"
#include "deformation.hpp"


#include <pybind11/pybind11.h>
namespace py = pybind11;



std::unordered_set<int> get_restricted_neighbors(int start, CellComplex &comp, std::unordered_set<int> &restricted, int max_dist) {
    
    auto cmp = [](const std::pair<int, int> &lhs, const std::pair<int, int> &rhs) {
        return lhs > rhs;
    };
    
    std::priority_queue< std::pair<int, int>, 
        std::vector<std::pair<int, int> >, decltype(cmp)> PQ(cmp);
    
    PQ.emplace(0, start);
    
    std::unordered_map<int, int> dists;
    
    while(!PQ.empty()) {
        
        auto top = PQ.top();
        PQ.pop();
        
        int d = top.first;
        int vi = top.second;
        
        if(!dists.count(vi) || dists[vi] > d) {
            dists[vi] = d;
        } else {
            continue;
        }
        
        for(auto ei: comp.get_cofacets(vi)) {
            if(restricted.count(ei)) {
                for(auto vj: comp.get_facets(ei)) {
                    if(vj != vi && d < max_dist) {
                        PQ.emplace(d+1, vj);
                    }
                }
            }
        }
    }
    
    std::unordered_set<int> neigh;
    for(auto pair: dists) {
        neigh.insert(pair.first);
    }
    return neigh;
    
}



// template <int DIM> CellComplex shrink_alpha_complex(CellComplex &comp, Filtration &filt, int max_dist, double threshold=0.0, bool verbose=false) {
    
    
//     auto cmp = [&filt](const int lhs, const int rhs) {
//         return filt.get_order(lhs) < filt.get_order(rhs);
//     };
        
//     std::vector<int> cell_order(comp.ndcells[1]);
//     std::iota(cell_order.begin(), cell_order.end(), comp.dcell_range[1].first);
//     std::sort(cell_order.begin(), cell_order.end(), cmp);
    
    
//     std::unordered_set<int> restricted;
//     std::unordered_set<int> incomplete;
    
//     for(int i = 0; i < comp.ndcells[0]; i++) {
//         incomplete.insert(i);
//     }
    
//     int c;
//     for(int i = 0; i < comp.ndcells[1]; i++) {
//         c = cell_order[i];
        
        
//         if(verbose && restricted.size() % 1000 == 0) {
//             py::print(restricted.size(), incomplete.size(), filt.get_func(c), py::arg("flush")=true);
//         }
        
//         restricted.insert(c);
        
//         if(filt.get_func(c) < threshold) {
//             continue;
//         }
                
//         std::vector<int> complete;
//         for(int vi: incomplete) {
                        
//             auto neigh = get_restricted_neighbors(vi, comp, restricted, max_dist);
//             // Includes vi
//             if(neigh.size() >= DIM+1) {
//                 complete.push_back(vi);
//             }
            
//         }
        
//         for(int vi: complete) {
//             incomplete.erase(vi);
//         }
        
//         if(incomplete.empty() && filt.get_func(c) > threshold) {
//             break;
//         }
        
//     }
    
//     std::vector<int> rem_cells;
//     for(int i = comp.dcell_range[1].first; i < comp.ncells; i++) {
//         if(filt.get_order(i) > filt.get_order(c)) {
//             rem_cells.push_back(i);
//         }
//     }
    
//     if(verbose) {
//         py::print("Final alpha", filt.get_func(c), py::arg("flush")=true);
//     }
    
    
//     return prune_cell_complex(rem_cells, comp);
    

// }


template <int DIM> CellComplex shrink_alpha_complex(CellComplex &comp, Embedding<DIM> &embed) {
    
    std::vector<int> edge_order(comp.ndcells[1]);
    std::iota(edge_order.begin(), edge_order.end(), comp.dcell_range[1].first);
    
    std::vector<double> eq_lengths(comp.ndcells[1]);
    for(int c = comp.dcell_range[1].first; c < comp.dcell_range[1].second; c++) {
        
        auto facets = comp.get_facets(c);
        int vi = facets[0];
        int vj = facets[1];
        
        DVec bvec = embed.get_diff(embed.get_vpos(vj), embed.get_vpos(vi));
        
//         py::print(c, comp.get_label(c), bvec.norm());
        
        eq_lengths[comp.get_label(c)] = bvec.norm();
        
    }
    
    auto cmp = [&comp, &eq_lengths](const int &lhs, const int &rhs) {
        return eq_lengths[comp.get_label(lhs)] > eq_lengths[comp.get_label(rhs)];
    };
    
    std::sort(edge_order.begin(), edge_order.end(), cmp);
    
//     for(int i = 0; i < comp.ndcells[1]; i++) {
//         py::print(i, edge_order[i], eq_lengths[comp.get_label(edge_order[i])]);
//     }
        
    std::vector<int> rem_cells;
    for(int i = 0; i < comp.ndcells[1]; i++) {
        int c = edge_order[i];
        
        rem_cells.push_back(c);
        
        CellComplex comp_tmp = prune_cell_complex(rem_cells, comp);
        
        auto betti = calc_betti_numbers(comp_tmp);
        
//         py::print(i, c, eq_lengths[comp.get_label(c)], betti);
        
        if(betti[0] > 1 || betti[comp.dim-1] > 0) {
            rem_cells.pop_back();
            break;
        }
        
        
    }
    
    
    return prune_cell_complex(rem_cells, comp);
    
    
}







template <int DIM> XVec transform_coords(RXVec X, RDMat F, RDVec u) {
    
    XVec x = XVec::Zero(X.size());
    
    for(int vi = 0; vi < X.size() / DIM; vi++) {
        x.segment<DIM>(DIM*vi) = F * X.segment<DIM>(DIM*vi) + u;
    }
    
    return x;
    
}


template <int DIM> DVec calc_com(RXVec X) {
    
    DVec com = DVec::Zero();
    
    for(int vi = 0; vi < X.size() / DIM; vi++) {
        com += X.segment<DIM>(DIM*vi);
    }
    
    com /= X.size() / DIM;
    
    return com;
    
}


// template <int DIM> double calc_rmsd(RXVec X1, RXVec X2) {
    
//     DMat X = DMat::Zero();
//     DMat Y = DMat::Zero();
    
//     int vi = 0;

//     DVec O1 = X1.segment<DIM>(DIM*vi);
//     DVec O2 = X2.segment<DIM>(DIM*vi);


//     for(int j = 1; j < X1.size() / DIM; j++) {
        
//         int vj = j;

        
//         DVec dX = X1.segment<DIM>(DIM*vj) - O1;
//         DVec dx = X2.segment<DIM>(DIM*vj) - O2;
        
//         X += dX * dX.transpose();
//         Y += dx * dX.transpose();


//     }
    
//     DMat F = Y * X.inverse();
    
//     DMat R;
//     std::tie(R, std::ignore) = decompose_def_grad<DIM>(F, false);
    
//     double rmsd = 0.0;
//     for(int j = 1; j < X1.size() / DIM; j++) {
            
//         int vj = j;
        
//         DVec dX = X1.segment<DIM>(DIM*vj) - O1;
//         DVec dx = X2.segment<DIM>(DIM*vj) - O2;

//         rmsd += (R*dX - dx).squaredNorm();

//     }
    
//     rmsd /= (X1.size() / DIM-1);
    
//     return sqrt(rmsd);
    
// }



template <int DIM> std::tuple<std::vector<int>,
std::map<std::vector<int>, std::vector<int> >, 
std::map<std::vector<int>, std::vector<std::vector<int> > > > get_neighbor_grid(double grid_length, Embedding<DIM> &embed) {



    std::map<std::vector<int>, std::vector<int> > grid;
    
    DVec L = embed.box_mat.diagonal();    
    
    std::vector<int> grid_size(DIM);
    for(int d = 0; d < DIM; d++) {
        grid_size[d] = int(L(d) / grid_length);
    }
    
    for(int vi = 0; vi < embed.NV; vi++) {
        
        std::vector<int> index(DIM);
        DVec pos = embed.get_pos(vi);
        for(int d = 0; d < DIM; d++) {            
            index[d] = int(pos(d) / L(d) * grid_size[d]);     
        }
        
        grid[index].push_back(vi);
        
    }
    
    
        
    std::map<std::vector<int>, std::vector<std::vector<int> > > grid_connections;
    for(auto gi: grid) {
        for(auto gj: grid) {
            
            DiVec diff(DIM);
            for(int d = 0; d < DIM; d++) {
                diff(d) = gi.first[d] - gj.first[d];
            }
            
            diff = diff.cwiseAbs();
            
            if(diff.maxCoeff() <= 1) {
                grid_connections[gi.first].push_back(gj.first);
            }
            
        }
    }
    
    return std::make_tuple(grid_size, grid, grid_connections);
    
    
}




template <int DIM> std::vector<int> get_grid_index(int vi, Embedding<DIM> &embed, std::vector<int> &grid_size) {
    
    DVec L = embed.box_mat.diagonal();    
    
    std::vector<int> index(DIM);
    DVec pos = embed.get_pos(vi);
    for(int d = 0; d < DIM; d++) {
        index[d] = int(pos(d) / L(d) * grid_size[d]);
    }
    
    return index;
    
    
}

template <int DIM> std::vector<int> get_grid_neighbors(int vi, Embedding<DIM> &embed, double max_dist, std::tuple<std::vector<int>,
                                                                                std::map<std::vector<int>, std::vector<int> >, 
                                                                                std::map<std::vector<int>, std::vector<std::vector<int> > > > &grid_info) {
    
    
    std::vector<int> grid_size;
    std::map<std::vector<int>, std::vector<int> > grid;
    std::map<std::vector<int>, std::vector<std::vector<int> > > grid_connections;
    std::tie(grid_size, grid, grid_connections) = grid_info;
    
    auto index = get_grid_index<DIM>(vi, embed, grid_size);
    
    DVec vposi = embed.get_vpos(vi);
        
    std::vector<int> verts;

    // Find neighbors
    for(auto gi: grid_connections[index]) {
        for(int vj: grid[gi]) {
            DVec bvec = embed.get_diff(vposi, embed.get_vpos(vj));

            if(bvec.norm() <= max_dist) {
                verts.push_back(vj);
            }
        }
    }

    verts.erase(std::remove(verts.begin(), verts.end(), vi), verts.end());
    verts.insert(verts.begin(), vi);
    
    return verts;
    
}


template <int DIM> double calc_hinge_overlap(std::map<int, std::set<int> > &sectors, RXVec disp, Embedding<DIM> &embed, bool linear=true) {
    
    XVec disp_pred = XVec::Zero(DIM*embed.NV);
    
    for(auto pair: sectors) {
        auto sector = pair.second;
        DVec xcm, ucm;
        DMat F, R;
        std::tie(xcm, ucm, F) = calc_bulk_motion<DIM>(sector, disp, embed, linear);
        std::tie(R, std::ignore) = decompose_def_grad<DIM>(F, linear);

        for(auto vi: sector) {
            disp_pred.segment<DIM>(DIM*vi) = (R - DMat::Identity()) * (embed.get_pos(vi) - xcm) + ucm;
        }
        
    }

    
    return disp.dot(disp_pred) / disp.norm() / disp_pred.norm();
    
}


template <int DIM> Embedding<DIM> get_net_embed(RXVec X) {
    
    int NV = X.size() / DIM;    
        
    XVec Lmin = XMatMap(X.data(), NV, DIM).rowwise().minCoeff();
    XVec Lmax = XMatMap(X.data(), NV, DIM).rowwise().maxCoeff();
    
    double L = (Lmax - Lmin).maxCoeff();

    DMat box_mat = L * DMat::Identity();

    XVec vert_pos = XVec::Zero(DIM*NV);
    
    for(int i = 0; i < NV; i++) {
        vert_pos.segment<DIM>(DIM*i) =(X.segment<DIM>(DIM*i) - Lmin) / L;
    }
    
    return Embedding<DIM>(NV, vert_pos, box_mat, false);
    
}


template <int DIM> std::vector<double> calc_err(std::function<double(RXVec, Embedding<DIM>&)> f, RXVec disp, Embedding<DIM> &embed, RXVec sigma_ref, RXVec sigma_def, int n_iters) {
    
    std::default_random_engine generator;
    std::normal_distribution<double> normal(0.0,1.0);
    
    XVec deltaX_ref = XVec::Zero(DIM*embed.NV);
    XVec deltaX_def = XVec::Zero(DIM*embed.NV);
    
    std::vector<double> q;
        
    XVec X_ref = XVec::Zero(DIM*embed.NV);
    for(int vi = 0; vi < embed.NV; vi++) {
        X_ref.segment<DIM>(DIM*vi) = embed.get_pos(vi);
    }
      
    for(int n = 0; n < n_iters; n++) {
                
        for(int i = 0; i < DIM*embed.NV; i++) {
            deltaX_ref(i) = normal(generator) * sigma_ref(i/DIM);
            deltaX_def(i) = normal(generator)* sigma_def(i/DIM);
        }
    
        XVec X_ref_prime = X_ref+deltaX_ref;
        auto embed_prime = get_net_embed<DIM>(X_ref_prime);
        
        XVec disp_prime = disp + deltaX_def-deltaX_ref;
        
        q.push_back(f(disp_prime, embed_prime));  
        
    }
    
    return q;
    
}

template <int DIM> std::vector<double> calc_lrmsd_err(int vi, RXVec disp, Embedding<DIM> &embed, SpacePartition<DIM> &part, double max_dist, RXVec sigma_ref, RXVec sigma_def, int n_iters, bool linear=true, bool weighted=false) {
    
    auto verts = part.get_neighbors(vi, max_dist);
   
    auto f = [vi, &verts, max_dist, linear, weighted](RXVec disp_prime, Embedding<DIM> &embed_prime) {
        
        XVec weights = XVec::Ones(verts.size());
        weights(0) = 0;
        if(weighted) {
            for(std::size_t j = 1; j < verts.size(); j++) {
                DVec bvec = embed_prime.get_diff(vi, verts[j]);
                weights(j) = exp(-bvec.squaredNorm() / pow(max_dist/2, 2.0) / 2);
            }
        }
        
        return calc_rmsd<DIM>(verts, disp_prime, embed_prime, weights, linear);
    };
    
    return calc_err<DIM>(f, disp, embed, sigma_ref, sigma_def, n_iters);

}


template <int DIM> std::vector<double> calc_hinge_overlap_err(std::map<int, std::set<int> > &sectors, RXVec disp, Embedding<DIM> &embed, RXVec sigma_ref, RXVec sigma_def, int n_iters, bool linear=true) {
    
    auto f = [&sectors, linear](RXVec disp_prime, Embedding<DIM> &embed_prime) {
        return calc_hinge_overlap<DIM>(sectors, disp_prime, embed_prime, linear);            
    };
    
    return calc_err<DIM>(f, disp, embed, sigma_ref, sigma_def, n_iters);

}









// template <int DIM> std::tuple<double, double> calc_rmsd_err(std::vector<int> &verts, RXVec disp, RXVec sigma, int n_iters, Embedding<DIM> &embed, double max_dist, bool linear=true) {
    
//     std::default_random_engine generator;
//     std::normal_distribution<double> normal(0.0,1.0);
    
//     XVec rmsd = XVec::Zero(embed.NV);

//     for(int n = 0; n < n_iters; n++) {
        
//         XVec delta = XVec::Zero(disp.size());
        
//         for(int i = 0; i < disp.size(); i++) {
//             delta(i) = normal(generator);
//         }
        
//         XVec disp_prime = disp + delta.cwiseProduct(sigma);
    
//         rmsd(n) = calc_rmsd<DIM>(verts, disp_prime, embed, max_dist, linear);
        
//     }

//     double rmsd_mean = rmsd.sum() / n_iters;
//     double rmsd_std = sqrt((rmsd - rmsd_mean*XVec::Ones(n_iters)).squaredNorm() / n_iters);

//     return std::make_tuple(rmsd_mean, rmsd_std);

// }



// template <int DIM> std::tuple<XVec, XVec> calc_local_rmsd_err(RXVec disp, RXVec sigma, int n_iters, Embedding<DIM> &embed, double max_dist, bool linear=true) {
    
//     auto grid_info = get_neighbor_grid<DIM>(max_dist, embed);
    
//     XVec lrmsd = XVec::Zero(embed.NV);
//     XVec lrmsd_err = XVec::Zero(embed.NV);

//     for(int vi = 0; vi < embed.NV; vi++) {
        
//         auto verts = get_grid_neighbors<DIM>(vi, embed, max_dist, grid_info);
        
//         std::tie(lrmsd(vi), lrmsd_err(vi)) = calc_rmsd_err<DIM>(verts, disp, embed, max_dist, linear);
        
//     }

//     return std::make_tupele(lrmsd, lrmsd_err);

// }




// template <int DIM> std::tuple<double, double, double> calc_hinge_overlap_err(std::set<int> &sector1, std::set<int> &sector2, RXVec disp, RXVec sigma, int n_iters, Embedding<DIM> &embed, double max_dist, bool linear=true) {
    
//     std::default_random_engine generator;
//     std::normal_distribution<double> normal(0.0,1.0);
    
//     double overlap_true = calc_hinge_overlap<DIM>(sector1, sector2, disp, embed, max_dist, linear);
    
//     XVec overlap = XVec::Zero(embed.NV);

//     for(int n = 0; n < n_iters; n++) {
        
//         XVec delta = XVec::Zero(disp.size());
        
//         for(int i = 0; i < disp.size(); i++) {
//             delta(i) = normal(generator);
//         }
        
//         XVec disp_prime = disp + delta.cwiseProduct(sigma);
    
//         overlap(n) = calc_hinge_overlap<DIM>(sector1, sector2, disp_prime, embed, max_dist, linear);       
//     }

//     double overlap_mean = overlap.sum() / n_iters;
//     double overlap_std = sqrt((overlap- overlap_mean*XVec::Ones(n_iters)).squaredNorm() / n_iters);

//     return std::make_tuple(overlap_true, overlap_mean, overlap_std);

// }



template <int DIM> XVec calc_local_strain(RXVec disp, Embedding<DIM> &embed, double max_dist, bool linear=true) {
    
    auto grid_info = get_neighbor_grid<DIM>(max_dist, embed);
    
    XVec strain = XVec::Zero(embed.NV);

    for(int vi = 0; vi < embed.NV; vi++) {
        
        auto verts = get_grid_neighbors<DIM>(vi, embed, max_dist, grid_info);
        
        DMat F;
        std::tie(F, std::ignore) = calc_def_grad<DIM>(verts, disp, embed, false);
        
        DMat eps = def_grad_to_strain(F, linear);
        
        strain(vi) = eps.norm();

    }

    return strain;
    
}

template <int DIM> std::tuple<DMat, Eigen::Matrix<double, DIM, DIM*DIM>, double> calc_def_grad_order2(std::vector<int> &verts, RXVec disp, Embedding<DIM> &embed, bool calc_D2min=false) {
    
    // Formulas and notation taken from:
    // Zimmerman, J. A., Bammann, D. J., & Gao, H. (2009). Deformation gradients for continuum mechanical analysis of atomistic simulations. International Journal of Solids and Structures, 46(2), 238–253.
    
    
    # define DIM2 DIM*DIM
    # define D2Vec Eigen::Matrix<double, DIM2, 1>
    # define DD2Mat Eigen::Matrix<double, DIM, DIM2>
    # define D2DMat Eigen::Matrix<double, DIM2, DIM>
    # define D2D2Mat Eigen::Matrix<double, DIM2, DIM2>
    
    DMat eta = DMat::Zero();
    DD2Mat xi = DD2Mat::Zero();
    D2D2Mat phi = D2D2Mat::Zero();
    
    
    DMat omega = DMat::Zero();
    DD2Mat nu = DD2Mat::Zero();

    
  
    
    int vi = verts[0];

    DVec O = embed.get_vpos(vi);
    DVec uO = disp.segment<DIM>(DIM*vi);

    // Assemble matrices
    for(std::size_t j = 1; j < verts.size(); j++) {
        
        int vj = verts[j];

        DVec dX = embed.get_diff(O, embed.get_vpos(vj));
        DVec dx = dX + disp.segment<DIM>(DIM*vj) - uO;
        
        DMat dXdX = dX*dX.transpose();
        
        D2Vec dXdX_vec = Eigen::Map<D2Vec> (dXdX.data());
        
//         D2Vec dXdX_vec = D2Vec::Zero();
//         for(int d1 = 0, k = 0; d1 < DIM; d1++) {
//             for(int d2 = d1; d2 < DIM; d2++, k++) {
//                 dXdX_vec(k) = dXdX(d1, d2);
//             }
//         }
  
        eta += dXdX;
        xi += dX * dXdX_vec.transpose();
        phi += dXdX_vec * dXdX_vec.transpose();
        
              
        omega += dx*dX.transpose();
        nu += dx * dXdX_vec.transpose();
        
        
    }    
    
//     py::print("verts", verts.size());
    
//     py::print("eta", eta);
//     py::print("xi", xi);
//     py::print("phi", phi);
    
//     py::print("omega", omega);
//     py::print("nu", nu);
    
    DMat eta_inv = eta.inverse();
    D2D2Mat zeta = phi - xi.transpose() * eta_inv * xi;
    
    D2D2Mat zeta_inv = zeta.completeOrthogonalDecomposition().pseudoInverse();
    
//     py::print("eta^(-1)", eta_inv);
    
//     py::print("xi^T eta^(-1) xi", xi.transpose() * eta_inv * xi);
    
//     py::print("zeta", zeta);

    
//     Eigen::SelfAdjointEigenSolver<D2D2Mat > eigensolver(zeta);
    
//     py::print(eigensolver.eigenvalues());
    
    
    
//     py::print("zeta^-1", zeta.inverse());
    
    DD2Mat H = 2 * (nu - omega * eta_inv * xi) * zeta_inv;
    
//     D2DMat y = 2 * (nu - omega * eta_inv * xi).transpose();
//     DD2Mat H = zeta.completeOrthogonalDecomposition().solve(y).transpose();
    
        
    DMat F = (omega - 0.5 * H * xi.transpose()) * eta_inv;
    
//     py::print("F", F);
//     py::print("H", H);
    
//     py::print("F^T H", F.transpose() * H);
    
        
    double D2min = 0.0;
    
    if(calc_D2min) {
        for(std::size_t j = 1; j < verts.size(); j++) {
            
            int vj = verts[j];

            DVec dX = embed.get_diff(O, embed.get_vpos(vj));
            DVec dx = dX + disp.segment<DIM>(DIM*vj) - uO;
            
            DMat dXdX = dX*dX.transpose();
            
            D2Vec dXdX_vec = Eigen::Map<D2Vec> (dXdX.data());
        
            D2min += (F*dX + H*dXdX_vec - dx).squaredNorm();

        }
        
        D2min /= (verts.size()-1);
    }
    
    
    return std::make_tuple(F, H, D2min);
    
    
}


template <int DIM> std::tuple<XVec, XVec > calc_local_strain_order2(RXVec disp, Embedding<DIM> &embed, double max_dist, bool linear=true) {
        
    
    auto grid_info = get_neighbor_grid<DIM>(max_dist, embed);
    
    XVec strain = XVec::Zero(embed.NV);
    XVec curv = XVec::Zero(embed.NV);

    for(int vi = 0; vi < embed.NV; vi++) {
        
        auto verts = get_grid_neighbors<DIM>(vi, embed, max_dist, grid_info);
        
        DMat F;
        Eigen::Matrix<double, DIM, DIM*DIM> H;
        std::tie(F, H, std::ignore) = calc_def_grad_order2(verts, disp, embed, false);
        
        DMat eps = def_grad_to_strain(F, linear);
        Eigen::Matrix<double, DIM*DIM, DIM*DIM> K = H.transpose() * H;
        
        strain(vi) = eps.norm();
        curv(vi) = K.norm();

    }

    return std::make_tuple(strain, curv);
    
}

void merge_basins(std::vector<int> &saddles, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp) {
    
    for(auto s: saddles) {
     
//             py::print("Saddle", i, j, s, filt.get_func(s), py::arg("flush")=true);

        auto morse_boundary = find_morse_boundary(s, V, comp, false, comp.oriented);

        std::vector<int> mverts;

        for(auto trip: morse_boundary) {
            int c, k;
            std::tie(c, k, std::ignore) = trip;

            if(k % 2 == 0) {
                py::print("Error", c, k);
            }

            mverts.push_back(c);

//                 py::print("vert", c, filt.get_func(c));

        }

        if(mverts.size() != 2) {
            py::print("Error", mverts, mverts.size());
        }


        auto pers_cmp = [s, &filt](const int &lhs, const int &rhs) {

            return std::abs(filt.get_func(s)-filt.get_func(lhs)) 
                < std::abs(filt.get_func(s)-filt.get_func(rhs));
        };

        int min_vert = *std::min_element(mverts.begin(), mverts.end(), pers_cmp);

//             py::print(filt.get_func(min_vert), filt.get_func(s), std::abs(filt.get_func(s)-filt.get_func(min_vert)));

        std::pair<int, int> cpair(min_vert, s);
        cancel_close_pair(cpair, V, coV, comp);

    }

    
    
    
}


template <int DIM> XVec calc_basin_boundary_dists(std::unordered_map<int, std::unordered_set<int> > &basins, CellComplex &comp, Embedding<DIM> &embed) {
    
    
    // First find vertices at boundaires between basins
    std::unordered_set<int> boundary_verts;
    
    for(int c = comp.dcell_range[1].first; c < comp.dcell_range[1].second; c++) {
        
        auto verts = comp.get_facets(c);
        int vi = verts[0];
        int vj = verts[1];
        
        int basini = -1;
        int basinj = -1;
        for(auto pair: basins) {
            
            if(pair.second.count(vi)) {
                basini = pair.first;
            }
            
            if(pair.second.count(vj)) {
                basinj = pair.first;
            }
            
        }
        
        if(basini == -1 || basinj == -1) {
            py::print("Error: verts not in basins.");
        }
        
        if(basini != basinj) {
            boundary_verts.insert(vi);
            boundary_verts.insert(vj);
        }
        
    }
    
    
    XVec dist = XVec::Constant(comp.ndcells[0], -1);
    // Find minimum distance of each vertex from boundary
    for(int vi = comp.dcell_range[0].first; vi < comp.dcell_range[0].second; vi++) {
        
        if(boundary_verts.count(vi)) {
            dist(vi) = 0.0;
            
            continue;
        }
        
        for(auto vj: boundary_verts) {
            
            double l0 = embed.get_diff(embed.get_vpos(vi), embed.get_vpos(vj)).norm();
            
            if(dist(vi) == -1.0 || l0 < dist(vi)) {
                dist(vi) = l0;
            }
                        
        }
        
    }
    
    return dist;
    
    
    
}


template <int DIM> XVec calc_basin_com_dists(std::unordered_map<int, std::unordered_set<int> > &basins, CellComplex &comp, Embedding<DIM> &embed) {
     
    
    XVec dist = XVec::Constant(comp.ndcells[0], -1);
    
    for(auto pair: basins) {
        
        // Calculate center of mass
        DVec com = DVec::Zero();
        for(auto vi: pair.second) {
            
            com += embed.get_pos(vi);
            
        }
        com /= pair.second.size();
        
        
        for(auto vi: pair.second) {
            
            dist(vi) = (embed.get_pos(vi) - com).norm();
            
        }
        
    }
    
    return dist;
    
    
    
}






// template <int DIM> CellComplex get_contact_network(Embedding<DIM> &embed, RXVec rad) {
    
    
//     double max_dist = 2.5*rad.maxCoeff();
    
//     std::map<std::vector<int>, std::vector<int> > grid;
//     std::map<std::vector<int>, std::vector<std::vector<int> > > grid_connections;
    
//     std::tie(grid, grid_connections) = get_neighbor_grid<DIM>(max_dist, embed);
    
    
        
//     DVec L = embed.box_mat.diagonal();    
    
//     DiVec nbins;
//     for(int d = 0; d < DIM; d++) {
//         nbins(d) = int(L(d) / max_dist);
//     }
    
    
    
//     CellComplex comp(1, true, false);
//     for(int vi = 0; vi < embed.NV; vi++) {
//         std::vector<int> facets;
//         std::vector<int> coeffs;
//         comp.add_cell(vi, 0, facets, coeffs);
//     }
    
//     for(int vi = 0; vi < embed.NV; vi++) {
        
//         std::vector<int> index(DIM);
//         DVec pos = embed.get_pos(vi);
//         for(int d = 0; d < DIM; d++) {
//             index[d] = int(pos(d) / L(d) * nbins(d));
//         }
        
//         DVec vposi = embed.get_vpos(vi);
        
//         // Find neighbors
//         for(auto gi: grid_connections[index]) {
//             for(int vj: grid[gi]) {
                
//                 if(vj <= vi) {
//                     continue;
//                 }
                
//                 DVec bvec = embed.get_diff(vposi, embed.get_vpos(vj));
                
//                 if(bvec.norm() <= rad[vi] + rad[vj]) {
//                     std::vector<int> facets;
//                     facets.push_back(vi);
//                     facets.push_back(vj);
                    
//                     std::vector<int> coeffs;
//                     comp.add_cell(comp.ndcells[1], 1, facets, coeffs);
                    
//                 }
//             }
//         }
        
//     }
    
//     comp.construct_cofacets();
//     comp.make_compressed();
    
//     return comp;
    
    
// }




// std::vector<std::pair<int, int> > find_hinge_persistence_pairs(std::vector<std::pair<int, int> > &pairs, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, int n_basins=2, int min_size=1, bool reset=false, bool verbose=false) {
    
    
//     XiVec V_tmp = V;
//     XiVec coV_tmp = coV;
    
//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
            
//     std::set<std::pair<int, int> > unsorted;
//     for(auto pair: pairs) {
//         unsorted.insert(pair);
//     }
    
    
//     if(verbose) {
//         py::print("Pass:", 0, py::arg("flush")=true);
//         py::print("Unsorted Pairs:", unsorted.size(), py::arg("flush")=true);
//     }
    
    
    
    
    
    
//      // Find morse basins
//     auto mcomp = construct_morse_complex(V_tmp, comp);
//     auto mbasins = find_morse_basins(coV_tmp, mcomp, comp);

//     if(verbose) {
//         py::print("Basins", mbasins.size(), py::arg("flush")=true);
//     }  

//     // Calculate basin sizes
    
//     std::vector<int> basins;
//     std::vector<int> sizes;
//     std::vector<int> size_sort(mbasins.size());
    
//     auto scmp = [&sizes] (const int lhs, const int rhs) {
//         return sizes[lhs] > sizes[rhs];
//     };
    
//     for(auto pair: mbasins) {
//         basins.push_back(pair.first);
//         sizes.push_back(pair.second.size());
//     }
    
//     std::iota(size_sort.begin(), size_sort.end(), 0);
//     std::sort(size_sort.begin(), size_sort.end(), scmp);
    
    
    


//     bool analyze = true;
    
//     std::unordered_set<int> hinge_basins;
    
//     std::pair<int, int> prev_pair;
    
//     for(int n = 1; ; n++) {
        
        
//         if(analyze) {
            
//             analyze = false;
            
//             if(verbose) {
//                 py::print("Basins", mbasins.size(), py::arg("flush")=true);
//             }  
            
//             if(verbose) {
//                 if(mbasins.size() >= 5) {
//                     py::print("Basin sizes", sizes[size_sort[0]], sizes[size_sort[1]], sizes[size_sort[2]], 
//                               sizes[size_sort[3]], sizes[size_sort[4]], py::arg("flush")=true);
//                 } else if(mbasins.size() == 4) {
//                     py::print("Basin sizes", sizes[size_sort[0]], sizes[size_sort[1]], sizes[size_sort[2]], 
//                               sizes[size_sort[3]], py::arg("flush")=true);
//                 } else if(mbasins.size() == 3) {
//                     py::print("Basin sizes", sizes[size_sort[0]], sizes[size_sort[1]], sizes[size_sort[2]], 
//                               py::arg("flush")=true);
//                 } else if(mbasins.size() == 2) {
//                     py::print("Basin sizes", sizes[size_sort[0]], sizes[size_sort[1]], py::arg("flush")=true);
//                 }
//             }
            
//             bool recorded = false;
//             // If enough basins
//             if((int)mbasins.size() >= n_basins) {
                
//                 // If nth basin is larger than min_size 
//                 // Or haven't recorded basins yet and there are only n left
//                 if(sizes[size_sort[n_basins-1]] >= min_size 
//                    || (hinge_basins.empty() && (int)mbasins.size() == n_basins)) {
                    
//                     // Record basins
//                     hinge_basins.clear();
//                     for(int i = 0; i < n_basins; i++) {
//                         hinge_basins.insert(basins[size_sort[i]]);
//                     }
                    
//                     recorded = true;
                    
//                     py::print("Recorded", hinge_basins);
                    
//                 }

//             }
            
//             // If a pair was previously cancelled (n > 1)
//             // And basins have not been recorded this pass
//             if(n > 1 && !recorded) {
                
//                 // Follow simplification path of basins and replace (but don't delete)
//                 if(hinge_basins.count(prev_pair.first)) {
                    
//                     auto morse_boundary = find_morse_boundary(prev_pair.second, V_tmp, comp, false, comp.oriented);
//                     for(auto trip: morse_boundary) {
//                         int c = std::get<0>(trip);
//                         if(!hinge_basins.count(c)) {
//                             hinge_basins.erase(prev_pair.first);
//                             hinge_basins.insert(c);
                            
//                             py::print("Replaced", prev_pair.first, "with", c);
//                             py::print(hinge_basins);
//                         }
//                     }
                    
//                 }   
                
//             }
            
            
            
//         }
        
//         // Reset pairs with each pass
//         if(reset) {
//             // Clear priority queue of cancellable pairs
//             while(!cancel_pairs.empty()) {
//                 auto top = cancel_pairs.top();
//                 cancel_pairs.pop();
//                 auto cpair = top.second;
//                 unsorted.insert(cpair);
//             }  
            
//         }

//         // Identify cancellable pairs
//         std::vector<std::pair<int, int> > remove;
//         for(auto cpair: unsorted) {
        
//             auto morse_boundary = find_morse_boundary(cpair.second, V_tmp, comp, false, comp.oriented);
                     
//             for(auto trip: morse_boundary) {
//                 int c = std::get<0>(trip);

//                 if(c == cpair.first) {
                    
//                     cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
//                     remove.push_back(cpair);
//                     break;
//                 }
//             }
//         }
        
//         for(auto cpair: remove) {
//             unsorted.erase(cpair);
//         }
    
        
//          if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//             py::print("Unsorted Pairs:", unsorted.size(), py::arg("flush")=true);
             
//         }
    
//         if(!cancel_pairs.empty()) {
//             auto top = cancel_pairs.top();
//             cancel_pairs.pop();

//             auto t = top.first;
//             auto cpair = top.second;
            
//             cancel_close_pair(cpair, V_tmp, coV_tmp, comp);
            
//             if(verbose) {
//                 py::print("Threshold:", t, cpair, py::arg("flush")=true);
//             }
            
            
//             if(comp.get_dim(cpair.first) == 0) {
               
//                 analyze = true;
//                 prev_pair = cpair;
                
                
//                 auto morse_boundary = find_morse_boundary(cpair.second, V_tmp, comp, false, comp.oriented);
//                 int c = std::get<0>(morse_boundary[0]);
//                 mbasins[c].insert(mbasins[cpair.first].begin(), mbasins[cpair.first].end());
//                 mbasins.erase(cpair.first);
                
//                 if(verbose) {
//                     py::print("Joining", cpair.first, c, py::arg("flush")=true);
//                 }
                
//                 basins.clear();
//                 sizes.clear();
//                 size_sort.resize(mbasins.size());
//                 for(auto pair: mbasins) {
//                     basins.push_back(pair.first);
//                     sizes.push_back(pair.second.size());
//                 }

//                 std::iota(size_sort.begin(), size_sort.end(), 0);
//                 std::sort(size_sort.begin(), size_sort.end(), scmp);
                
                
                
//             }
            
//         } else {
//             break;
//         }
        
//     }
    
    
//     std::vector<std::pair<int, int> > hinge_pairs;
//     for(auto pair: pairs) {
//         if(hinge_basins.count(pair.first)) {
//             hinge_pairs.push_back(pair);
//         }
//     }

    
//     return hinge_pairs;
    
// }





// void simplify_morse_complex(std::vector<std::pair<int, int> > &pairs, RXiVec V, RXiVec coV, Filtration &filt, CellComplex &comp, bool reset=false, bool verbose=false) {
    
    
    
//     auto cmp = [](const std::pair<double, std::pair<int, int> > &lhs, const std::pair<double, std::pair<int, int> > &rhs) {
//         return lhs > rhs;
//     };
    
//     std::priority_queue<std::pair<double, std::pair<int, int> >, 
//         std::vector<std::pair<double, std::pair<int, int> > >, decltype(cmp)> cancel_pairs(cmp);
    
            
//     std::set<std::pair<int, int> > unsorted;
//     for(auto pair: pairs) {
//         unsorted.insert(pair);
//     }
    
    
//     if(verbose) {
//         py::print("Pass:", 0, py::arg("flush")=true);
//         py::print("Unsorted Pairs:", unsorted.size(), py::arg("flush")=true);
//     }

//     std::pair<int, int> hinge_pair;
    
//     for(int n = 1; ; n++) {
        
//         // Reset pairs with each pass
//         if(reset) {
//             // Clear priority queue of cancellable pairs
//             while(!cancel_pairs.empty()) {
//                 auto top = cancel_pairs.top();
//                 cancel_pairs.pop();
//                 auto cpair = top.second;
//                 unsorted.insert(cpair);
//             }  
            
//         }

        
//         std::vector<std::pair<int, int> > remove;
//         for(auto cpair: unsorted) {
        
//             std::vector<std::tuple<int, int, int> > morse_boundary = find_morse_boundary(cpair.second, V, comp, false, comp.oriented);
            
//             for(auto trip: morse_boundary) {
//                 int c, k;
//                 std::tie(c, k, std::ignore) = trip;

//                 if(c == cpair.first) {
                    
//                     cancel_pairs.emplace(std::abs(filt.get_func(cpair.second)-filt.get_func(cpair.first)), cpair);
//                     remove.push_back(cpair);
//                     break;
//                 }
//             }
//         }
        
//         for(auto cpair: remove) {
//             unsorted.erase(cpair);
//         }
    
        
//         if(verbose) {
//             py::print("Pass:", n, py::arg("flush")=true);
//             py::print("Cancellable Pairs:", cancel_pairs.size(), py::arg("flush")=true);
//             py::print("Unsorted Pairs:", unsorted.size(), py::arg("flush")=true);
             
//         }
    
//         if(!cancel_pairs.empty()) {
//             auto top = cancel_pairs.top();
//             cancel_pairs.pop();

//             auto t = top.first;
//             auto cpair = top.second;
            
            
//             if(verbose) {
//                 py::print("Cancelling:", cpair, t, py::arg("flush")=true);
//             }
            
            
//             cancel_close_pair(cpair, V, coV, comp);
            
//         } else {
//             break;
//         }
        
//     }
    
// }




#endif // PROTEIN_HPP

