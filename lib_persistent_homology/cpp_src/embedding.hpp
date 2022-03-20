#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "eigen_macros.hpp"
  
template <int DIM> class Embedding {
 
public:
    
    // Embedding informtaion for a collection of points
        
    // dimension
    const int dim;
    
protected:
    
    // vertex positions (undeformed and in unit box)
    XVec vert_pos;
    
public:
    // Number of vertices
    int NV;
    // box matrix (multiply by pos to get actual embedded position)
    DMat box_mat;
    // inverse box matrix
    DMat box_mat_inv;
    // boundary conditions
    bool periodic;
    
    Embedding(int NV, RXVec vert_pos, RDMat box_mat, bool periodic) :
        dim(DIM), vert_pos(vert_pos), NV(NV), box_mat(box_mat), box_mat_inv(box_mat.inverse()), periodic(periodic) {}
    
    // Get virtual position
    inline DVec get_vpos(int vi) {
        return vert_pos.segment<DIM>(DIM*vi);
    }
    
    // Get real position
    inline DVec get_pos(int vi) {
        return box_mat * vert_pos.segment<DIM>(DIM*vi);
    };
    
    void transform(RDMat trans, RDVec offset) {
        for(int vi = 0; vi < NV; vi++) {
            vert_pos.segment<DIM>(DIM*vi) = trans * vert_pos.segment<DIM>(DIM*vi) + offset;
        }
    }
    
    // Vector difference according to BCs
    // Takes in virtual coords
    // Return virtual coords
    inline DVec get_vdiff(DVec const &xi, DVec const &xj) {
        
        DVec dx;            
        
        dx = xj - xi;
        
        if(periodic) {
            
            for(int d = 0; d < DIM; d++) {
                if(std::fabs(dx(d)) > 0.5) {
                    dx(d) -= ((dx(d) > 0) - (dx(d) < 0));
                }
            }
            
        }
        
        return dx;
        
    }
    
    inline DVec get_vdiff(int vi, int vj) {

        DVec xi = get_vpos(vi);
        DVec xj = get_vpos(vj);
        
        return get_vdiff(xi, xj);
        
    }
    
    // Takes in virtual coords
    // Returns real coords
    inline DVec get_diff(DVec const &xi, DVec const &xj) {
        
        DVec dx;
            
        dx = get_vdiff(xi, xj);
        
        return box_mat * dx;
        
    }
    
    // Maybe implement version like this:
    // Or have one get_diff function with two template parameters (input virtual or real, output virtual or real)
    // Then in python side have get_vdiffv, get_vdiffr, get_rdiffv, get_rdiffr
    
//      template<bool VIRTUAL=true> inline DVec get_diff(DVec const &xi, DVec const &xj) {
        
//         DVec dx;
            
//         if(VIRTUAL) {
//             dx = get_vdiff(xi, xj);
//         } else {
//             dx = get_vdiff(box_mat_inv*xi, box_mat_inv*xj);
//         }
        
//         return box_mat * dx;
        
//     }
    
    inline DVec get_diff(int vi, int vj) {

        DVec xi = get_vpos(vi);
        DVec xj = get_vpos(vj);
        
        return get_diff(xi, xj);
        
    }
   
    //takes in real, returns real
    inline DVec get_diff_realpos(DVec const &xi, DVec const &xj) { 
        DMat inv = box_mat.inverse() ;
        return get_diff(inv*xi, inv*xj); 
    }
    
};

    
#endif // EMBEDDING_HPP
