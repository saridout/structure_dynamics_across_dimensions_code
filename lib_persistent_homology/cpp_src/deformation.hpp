#ifndef DEFORM_HPP
#define DEFORM_HPP



#include <vector>
#include <set>

#include "eigen_macros.hpp"
#include "embedding.hpp"
#include "cell_complex.hpp"
#include "search.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;



template <int DIM> std::tuple<DMat, double> calc_def_grad(std::vector<int> &verts, RXVec disp, Embedding<DIM> &embed, bool calc_D2min=false) {
    
    XVec weight = XVec::Ones(verts.size());
    weight(0) = 0;

    return calc_def_grad<DIM>(verts, disp, embed, weight, calc_D2min);
    
    
}


template <int DIM> std::tuple<DMat, double> calc_def_grad(std::vector<int> &verts, RXVec disp, Embedding<DIM> &embed, RXVec weight, bool calc_D2min=false) {
    

    DMat X = DMat::Zero();
    DMat Y = DMat::Zero();
    
    
    int vi = verts[0];

    DVec O = embed.get_vpos(vi);
    DVec uO = disp.segment<DIM>(DIM*vi);

    for(std::size_t j = 1; j < verts.size(); j++) {
        
        int vj = verts[j];

        DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

        DVec du = disp.segment<DIM>(DIM*vj) - uO;
        
        X += weight(j)*bvec * bvec.transpose();
        Y += weight(j)*(bvec + du) * bvec.transpose();


    }
    
    DMat F = Y * X.inverse();
        
    double D2min = 0.0;
    
    if(calc_D2min) {
        for(std::size_t j = 1; j < verts.size(); j++) {
            
            int vj = verts[j];

            DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

            DVec du = disp.segment<DIM>(DIM*vj) - uO;
            

            D2min += weight(j)*(F*bvec - (bvec+du)).squaredNorm();

        }
        
//         D2min /= (verts.size() - 1);
        D2min /= weight.sum();
    }
    
    
    return std::make_tuple(F, D2min);
    
    
}


template <int DIM> DMat def_grad_to_strain(DMat &F, bool linear=true) {
    
    DMat eps;
    if(linear) {
        eps = 0.5*(F + F.transpose()) - DMat::Identity();
    } else {
        eps = 0.5*(F.transpose() * F  - DMat::Identity());
    }
    
    return eps;
    
}


template <int DIM> std::tuple<DMat, DMat> decompose_def_grad(DMat &F, bool linear=true) {
    
    
    if(linear) {
        DMat dR = 0.5*(F - F.transpose());
        DMat eps = 0.5*(F + F.transpose()) - DMat::Identity();
        
        return std::make_tuple(dR, eps);
        
        
    } else {
        
        DMat C = F.transpose() * F;
        
        Eigen::SelfAdjointEigenSolver<DMat> eigensolver(C);
        
        DMat U = eigensolver.eigenvectors() * eigensolver.eigenvalues().cwiseSqrt().asDiagonal() * eigensolver.eigenvectors().transpose();
        
        DMat R = F * U.inverse();
        
        return std::make_tuple(R, U);
    }
    
}
                                                          
                                                          
template <int DIM> XVec calc_tri_strains(RXVec disp, CellComplex &comp, Embedding<DIM> &embed, bool linear=true) {


    XVec strains = XVec::Zero(comp.ndcells[DIM]);

    for(int c = comp.dcell_range[DIM].first; c < comp.ncells; c++) {

        auto vset = comp.get_faces(c, 0);

        std::vector<int> verts(vset.begin(), vset.end());
                
        DMat F;
        std::tie(F, std::ignore) = calc_def_grad<DIM>(verts, disp, embed, false);
        
        DMat eps  = def_grad_to_strain<DIM>(F, linear);
        
        strains[comp.get_label(c)] = eps.norm();
                        
    }    

    return strains;




}


template <int DIM> std::tuple<XVec, XVec> calc_delaunay_D2min_strain(RXVec disp, CellComplex &comp, Embedding<DIM> &embed, int max_dist=2, bool linear=true) {


    XVec D2min = XVec::Zero(comp.ndcells[0]);
    XVec strains = XVec::Zero(comp.ndcells[0]);

    for(int vi = 0; vi < comp.ndcells[0]; vi++) {

        auto vset = find_neighbors(vi, comp, max_dist, 0);
        vset.erase(vi);
        
        std::vector<int> verts(vset.begin(), vset.end());
        verts.insert(verts.begin(), vi);
        
        
        DMat F;
        std::tie(F, D2min[vi]) = calc_def_grad<DIM>(verts, disp, embed, true);
        DMat eps  = def_grad_to_strain<DIM>(F, linear);
        
        strains[vi] = eps.norm();

    }

    return std::make_tuple(D2min, strains);


}


template <int DIM> double calc_rmsd(std::vector<int> &verts, RXVec disp, Embedding<DIM> &embed, RXVec weight, bool linear=true) {
    
    
    double rmsd = 0.0;

        
    DMat F;
    std::tie(F, std::ignore) = calc_def_grad<DIM>(verts, disp, embed, weight, false);

    DMat R;
    std::tie(R, std::ignore) = decompose_def_grad<DIM>(F, linear);

    int vi = verts[0];
    
//     DVec O = embed.get_vpos(vi);
    DVec uO = disp.segment<DIM>(DIM*vi);

    for(std::size_t j = 1; j < verts.size(); j++) {

        int vj = verts[j];

//         DVec bvec = embed.get_diff(O, embed.get_vpos(vj));
        DVec bvec = embed.get_diff(vi, vj);

        DVec du = disp.segment<DIM>(DIM*vj) - uO;


        if(linear) {
            rmsd += weight(j)*(R*bvec - du).squaredNorm();
        } else {
            rmsd += weight(j)*(R*bvec - (bvec+du)).squaredNorm();
        }
        
    }

//     rmsd /= (verts.size()-1);
    rmsd /= weight.sum();
        

    return sqrt(rmsd);

}


template <int DIM> XVec calc_delaunay_local_rmsd(RXVec disp, CellComplex &comp, Embedding<DIM> &embed, int max_dist=2, bool linear=true) {

    
    XVec lrmsd = XVec::Zero(embed.NV);

    for(int vi = 0; vi < comp.ndcells[0]; vi++) {
        

        auto vset = find_neighbors(vi, comp, max_dist, 0);
        vset.erase(vi);
        
        std::vector<int> verts(vset.begin(), vset.end());
        verts.insert(verts.begin(), vi);
        
        
        XVec weights = XVec::Ones(verts.size());
        weights(0) = 0;
                
        lrmsd(vi) = calc_rmsd<DIM>(verts, disp, embed, weights, linear);
        
    }

    return lrmsd;

    
    
    

//     XVec D2min = XVec::Zero(comp.ndcells[0]);

//     for(int vi = 0; vi < comp.ndcells[0]; vi++) {

//         auto verts = find_neighbors(vi, comp, max_dist, 0);
//         verts.erase(vi);
        
//         std::vector<int> verts(verts.begin(), verts.end());
//         verts.insert(verts.begin(), vi);
        
        
//         DMat F;
//         std::tie(F, std::ignore) = calc_def_grad<DIM>(verts, disp, embed, false);
        
//         DMat dR  = 0.5*(F - F.transpose());
        
        
//         DVec O = embed.get_vpos(vi);
//         DVec uO = disp.segment<DIM>(DIM*vi);
        
//         for(std::size_t j = 1; j < verts.size(); j++) {
            
//             int vj = verts[j];

//             DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

//             DVec du = disp.segment<DIM>(DIM*vj) - uO;
            

//             D2min[vi] += (dR*bvec - du).squaredNorm();

//         }
        
//         D2min[vi] /= (verts.size()-1);
        
//     }

//     return D2min;


}




template <int DIM> XVec calc_local_rmsd(RXVec disp, Embedding<DIM> &embed, SpacePartition<DIM> &part, double max_dist, bool linear=true, bool weighted=false) {
    
//     auto grid_info = get_neighbor_grid<DIM>(max_dist, embed);
    
    XVec lrmsd = XVec::Zero(embed.NV);

    for(int vi = 0; vi < embed.NV; vi++) {
        
//         auto verts = get_grid_neighbors<DIM>(vi, embed, max_dist, grid_info);
        auto verts = part.get_neighbors(vi, max_dist);
        
        XVec weights = XVec::Ones(verts.size());
        weights(0) = 0;
        if(weighted) {
            for(std::size_t j = 1; j < verts.size(); j++) {
                DVec bvec = embed.get_diff(vi, verts[j]);
                weights(j) = exp(-bvec.squaredNorm() / pow(max_dist/3, 2.0) / 2);
            }
        }
                
        lrmsd(vi) = calc_rmsd<DIM>(verts, disp, embed, weights, linear);
        
    }

    return lrmsd;

}



template <int DIM> std::tuple<DVec, DVec, DMat> calc_bulk_motion(std::set<int> &verts, RXVec disp, Embedding<DIM> &embed, bool linear=true) {
    
    DVec xcm = DVec::Zero();
    DVec ucm = DVec::Zero();
    
    for(auto vi: verts) {
        xcm += embed.get_pos(vi);
        ucm += disp.segment<DIM>(DIM*vi);
    }
    
    xcm /= verts.size();
    ucm /= verts.size();
    
    DMat X = DMat::Zero();
    DMat Y = DMat::Zero();
    for(auto vi: verts) {
        DVec dxi = embed.get_pos(vi) - xcm;
        DVec dui = disp.segment<DIM>(DIM*vi) - ucm;
        X += dxi * dxi.transpose();
        Y += (dxi + dui) * dxi.transpose();
    }
    
    DMat F = Y * X.inverse();
    
    return std::make_tuple(xcm, ucm, F);
    
}




// Remember that the first vertex in each group is the reference point - need reference if there is non-affine motion
template <int DIM> std::tuple<XVec, XVec> calc_grouped_delaunay_D2min_strain(std::vector<std::vector<int> > &groups, RXVec disp, CellComplex &comp, Embedding<DIM> &embed, int max_dist=2, bool linear=true) {


    XVec D2min = XVec::Zero(groups.size());
    XVec strains = XVec::Zero(groups.size());

    for(std::size_t gi = 0; gi < groups.size(); gi++) {
        
        std::unordered_set<int> gset;
        
        for(int vi: groups[gi]) {

            auto vset = find_neighbors(vi, comp, max_dist, 0);
            gset.insert(vset.begin(), vset.end());
            
        }
        
        // Set first as reference vertex
        gset.erase(groups[gi][0]);

        std::vector<int> verts(gset.begin(), gset.end());
        verts.insert(verts.begin(), groups[gi][0]);


        DMat F;
        std::tie(F, D2min[gi]) = calc_def_grad<DIM>(verts, disp, embed, true);
        DMat eps  = def_grad_to_strain<DIM>(F, linear);

        strains[gi] = eps.norm();

    }

    return std::make_tuple(D2min, strains);


}


template <int DIM> XVec subtract_global_motion(RXVec disp, Embedding<DIM> &embed, bool linear = true) {
    
    
    
    DMat X = DMat::Zero();
    DMat Y = DMat::Zero();
    
    
    DVec xcm = DVec::Zero();
    DVec ucm = DVec::Zero();
    
    
    for(int vi = 0; vi < embed.NV; vi++) {
        
        xcm += embed.get_vpos(vi);
        ucm += disp.segment<DIM>(DIM*vi);
        
    }
    
    xcm /= embed.NV;
    ucm /= embed.NV;
    

    for(int vj = 0; vj < embed.NV; vj++) {
        
        DVec bvec = embed.get_diff(xcm, embed.get_vpos(vj));

        DVec du = disp.segment<DIM>(DIM*vj) - ucm;

        X += bvec * bvec.transpose();
        Y += (bvec + du) * bvec.transpose();


    }

    DMat F = Y * X.inverse();
    
    XVec u = XVec::Zero(disp.size());
    
    if(linear) {
        DMat dR;
        DMat eps;
        
        std::tie(dR, eps) = decompose_def_grad<DIM>(F, linear);
    } else {
        
        DMat R;
        DMat U;
                
        std::tie(R, U) = decompose_def_grad<DIM>(F, linear);
        
        for(int vi = 0; vi < embed.NV; vi++) {
            
            DVec bvec = embed.get_diff(xcm, embed.get_vpos(vi));
            
            u.segment<DIM>(DIM*vi) =  R.transpose()*(bvec + disp.segment<DIM>(DIM*vi) - ucm) - bvec;
        }
        
    }
        
        
    
    return u;
    
    
}
                                                          
                                                          


// template <int DIM> XVec calc_strains(RXVec disp, CellComplex &comp, Embedding<DIM> &embed, bool keep_rotations=false) {


//     XVec strains = XVec::Zero(comp.ndcells[DIM]);

//     for(int c = comp.dcell_range[DIM].first; c < comp.ncells; c++) {

//         auto vset = comp.get_faces(c, 0);

//         std::vector<int> verts(vset.begin(), vset.end());


//         int vi = verts[0];

//         DVec O = embed.get_vpos(vi);
//         DVec uO = disp.segment<DIM>(DIM*vi);

//         DMat X = DMat::Zero();
//         DMat Y = DMat::Zero();


//         for(int m = 0; m < DIM; m++) {

//             int vj = verts[1+m];

//             DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

//             DVec du = disp.segment<DIM>(DIM*vj) - uO;

//             X += bvec * bvec.transpose();
//             Y += du * bvec.transpose();

//         }

//         DMat F = Y * X.inverse();

//         DMat eps;
//         if(keep_rotations) {
//             eps = F;
//         } else {
//             eps = 0.5 * (F + F.transpose());
//         }

//         strains(comp.get_label(c)) = eps.norm();
                        
//     }    

//     return strains;




// }


// template <int DIM> XVec calc_stresses(RXVec disp, RXVec K, CellComplex &comp, Embedding<DIM> &embed) {

    
//     XVec stress = XVec::Zero(comp.ndcells[DIM]);

//     for(int c = comp.dcell_range[DIM].first; c < comp.dcell_range[DIM].second; c++) {

//         auto vset = comp.get_faces(c, 0);

//         std::vector<int> verts(vset.begin(), vset.end());


//         int vi = verts[0];

//         DVec O = embed.get_vpos(vi);
//         DVec uO = disp.segment<DIM>(DIM*vi);

//         DMat X = DMat::Zero();
//         DMat Y = DMat::Zero();


//         for(int m = 0; m < DIM; m++) {

//             int vj = verts[1+m];

//             DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

//             DVec du = disp.segment<DIM>(DIM*vj) - uO;

//             X += bvec * bvec.transpose();
//             Y += du * bvec.transpose();

//         }

//         DMat F = Y * X.inverse();
        
//         DMat sigma = DMat::Zero();
        
//         for(int e: comp.get_faces(c, 1)) {
//             auto everts = comp.get_facets(e);
            
//             int vi = everts[0];
//             int vj = everts[1];
            
//             DVec bvec = embed.get_diff(embed.get_vpos(vi), embed.get_vpos(vj));
//             double ell2 = bvec.squaredNorm();
            
//             double ext = bvec.transpose() * F * bvec;
            
//             sigma += K(comp.get_label(e)) / ell2 * ext * bvec * bvec.transpose();
            
            
//         }
        
//         stress(comp.get_label(c)) = sigma.norm();

        
//     }
    

//     return stress;

// }


// template <int DIM> XVec calc_voronoi_D2min(RXVec disp, CellComplex &comp, Embedding<DIM> &embed, int max_dist=2) {


//     XVec D2min = XVec::Zero(embed.NV);

//     for(int vi = 0; vi < comp.ndcells[0]; vi++) {

//         auto verts = find_neighbors(vi, comp, max_dist, 0);

//         verts.erase(vi);

//         DVec O = embed.get_vpos(vi);
//         DVec uO = disp.segment<DIM>(DIM*vi);

//         DMat X = DMat::Zero();
//         DMat Y = DMat::Zero();


//         for(auto vj: verts) {

//             DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

//             DVec du = disp.segment<DIM>(DIM*vj) - uO;

//             X += bvec * bvec.transpose();
//             Y += du * bvec.transpose();
            
            

//         }

//         DMat F = Y * X.inverse();


//         for(auto vj: verts) {

//             DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

//             DVec du = disp.segment<DIM>(DIM*vj) - uO;
            

//             D2min(vi) += (du - F*bvec).squaredNorm();

//         }
        
//         D2min(vi) /= verts.size();

//     }

//     return D2min;


// }


template <int DIM> XVec calc_voronoi_D2min(RXVec disp, CellComplex &comp, Embedding<DIM> &embed, std::vector<bool> &is_contact) {


    XVec D2min = XVec::Zero(embed.NV);

    for(int vi = comp.dcell_range[0].first; vi < comp.dcell_range[0].second; vi++) {
        
        std::unordered_set<int> verts;
        
        for(auto ei: comp.get_cofacets(vi)) {
            if(is_contact[comp.get_label(ei)]) {
                auto everts = comp.get_facets(ei);
                int vj = (everts[0] == vi) ? everts[1] : everts[0];
               
                verts.insert(vj);
            }
        }

        DVec O = embed.get_vpos(vi);
        DVec uO = disp.segment<DIM>(DIM*vi);

        DMat X = DMat::Zero();
        DMat Y = DMat::Zero();


        for(auto vj: verts) {

            DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

            DVec du = disp.segment<DIM>(DIM*vj) - uO;

            X += bvec * bvec.transpose();
            Y += du * bvec.transpose();
                        

        }

        DMat F = Y * X.inverse();


        for(auto vj: verts) {

            DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

            DVec du = disp.segment<DIM>(DIM*vj) - uO;
            

            D2min(vi) += (du - F*bvec).squaredNorm();

        }
        
        D2min(vi) /= verts.size();

    }

    return D2min;


}




// template <int DIM> std::tuple<XVec, XVec> calc_delaunay_D2min_strain(RXVec disp, CellComplex &comp, Embedding<DIM> &embed, int max_dist=2, bool linear=true) {


//     XVec D2min = XVec::Zero(comp.ndcells[0]);
//     XVec strains = XVec::Zero(comp.ndcells[0]);

//     for(int vi = 0; vi < comp.ndcells[0]; vi++) {

//         auto verts = find_neighbors(vi, comp, max_dist, 0);

//         verts.erase(vi);

//         DVec O = embed.get_vpos(vi);
//         DVec uO = disp.segment<DIM>(DIM*vi);

//         DMat X = DMat::Zero();
//         DMat Y = DMat::Zero();


//         for(auto vj: verts) {

//             DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

//             DVec du = disp.segment<DIM>(DIM*vj) - uO;

//             X += bvec * bvec.transpose();
//             Y += du * bvec.transpose();
                        

//         }

//         DMat F = Y * X.inverse();


//         for(auto vj: verts) {

//             DVec bvec = embed.get_diff(O, embed.get_vpos(vj));

//             DVec du = disp.segment<DIM>(DIM*vj) - uO;
            

//             D2min(vi) += (du - F*bvec).squaredNorm();

//         }
        
//         D2min(vi) /= verts.size();
        
//         DMat eps = 0.5 * (F + F.transpose());
//         strains(vi) = eps.norm();

//     }

//     return std::make_tuple(D2min, strains);


// }



// Calculate condition numbers of triangulation element stiffness tensors
template <int DIM> XVec calc_flatness(CellComplex &comp, Embedding<DIM> &embed) {


    XVec flatness = XVec::Zero(comp.ndcells[DIM]);

    for(int c = comp.dcell_range[DIM].first; c < comp.ncells; c++) {

        auto vset = comp.get_faces(c, 0);

        std::vector<int> verts(vset.begin(), vset.end());

        // Each column is an altitude vector pointing from a (d-1)-face to the opposite vertex
        XMat altitudes(DIM, DIM+1);
        
        for(int i = 0; i < DIM+1; i++) {
            
            int vi = verts[i];
            
            std::vector<int> fverts;
            for(int j = 0; j < DIM+1; j++) {
                if(j != i) {
                    fverts.push_back(verts[j]);
                }
            }
            
            // Calculate positions of all verts relative to one vert
            XMat cross_mat(DIM, DIM-1);
            
            DVec O = embed.get_vpos(fverts[0]);
            for(int m = 0; m < DIM-1; m++) {
                cross_mat.col(m) = embed.get_diff(O, embed.get_vpos(fverts[1+m]));
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
            
            normal.normalize();
            
            // Calculate altitude vector
            DVec u = embed.get_diff(O, embed.get_vpos(vi));
            DVec a = normal.dot(u) * normal;
            a /= a.squaredNorm();
            altitudes.col(i) = a;
            
        }
        
        // Calculate element stiffness matrix
        XMat X = altitudes.transpose() * altitudes;
        
        
        Eigen::SelfAdjointEigenSolver<DMat> esolver(X);
        
        DVec evals = esolver.eigenvalues();

        flatness(comp.get_label(c)) = evals[DIM-1] / evals[0];
                
        
    }
        

    return flatness;




}











#endif //DEFORM_HPP