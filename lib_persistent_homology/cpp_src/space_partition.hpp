#ifndef SPACE_PART_HPP
#define SPACE_PART_HPP

#include "eigen_macros.hpp"
#include "embedding.hpp"

#include <vector>
#include <map>
#include <numeric>
#include "math.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

  
template <int DIM> class SpacePartition {
    
    public:
        
//     std::vector<int> get_neighbors(int vi, double max_dist) {
//         return std::vector<int>(1, vi);
//     };
    
    virtual std::vector<int> get_neighbors(int vi, double max_dist) = 0;
    
};

template <int DIM> class NeighborGrid: public SpacePartition<DIM> {
    
    private:
    
    Embedding <DIM> &embed;
    double grid_length;
    std::vector<int> grid_size;
    std::map<std::vector<int>, std::vector<int> > grid;
    std::map<std::vector<int>, std::vector<std::vector<int> > > grid_connections;
    
    public:
    
    NeighborGrid(Embedding<DIM> &embed, double grid_length) : embed(embed), grid_length(grid_length) {
        
        
        DVec L = embed.box_mat.diagonal();    

        grid_size.resize(DIM);
        for(int d = 0; d < DIM; d++) {
            grid_size[d] = int(L(d) / grid_length);
        }

        for(int vi = 0; vi < embed.NV; vi++) {
            auto index = get_grid_index(vi, embed, grid_size);
            grid[index].push_back(vi);

        }

        for(auto gi: grid) {
            for(auto gj: grid) {

                DiVec diff(DIM);
                for(int d = 0; d < DIM; d++) {

                    if(embed.periodic) {
                        diff(d) = (gi.first[d] - gj.first[d]) % grid_size[d];
                    } else {
                        diff(d) = abs(gi.first[d] - gj.first[d]);
                    }
                }

                if(diff.maxCoeff() <= 1) {
                    grid_connections[gi.first].push_back(gj.first);
                }

            }
        }
        
        
        
    }
    
    private:
    
    std::vector<int> get_grid_index(int vi, Embedding<DIM> &embed, std::vector<int> &grid_size)  {

        DVec L = embed.box_mat.diagonal();    

        std::vector<int> index(DIM);
        DVec pos = embed.get_pos(vi);
        for(int d = 0; d < DIM; d++) {
            index[d] = int(pos(d) / L(d) * grid_size[d]);
        }

        return index;

    }
    
    public:
    
    std::vector<int> get_neighbors(int vi, double max_dist) {
        
        auto index = get_grid_index(vi, embed, grid_size);

        std::vector<int> verts;

        // Find neighbors
        for(auto gi: grid_connections[index]) {
            for(int vj: grid[gi]) {
                DVec bvec = embed.get_diff(vi, vj);

                if(bvec.norm() <= max_dist) {
                    verts.push_back(vj);
                }
            }
        }

        verts.erase(std::remove(verts.begin(), verts.end(), vi), verts.end());
        verts.insert(verts.begin(), vi);

        return verts;

    }
    

};

// Test code ofr python

// NV = 9
     
// vert_pos = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4, 4, 3, 5, 2, 6, 1, 7, 0, 8])/8

// box_mat = np.identity(2)

// embed = phom.Embedding2D(NV, vert_pos, np.asfortranarray(box_mat), False)

// phom.KDTree2D(embed)

// template <int DIM> class KDTree: public SpacePartition<DIM> {
    
//     private:
    
//     Embedding <DIM> &embed;
//     // Tree whose elements refer to parent
//     // Root node points to itself
//     std::vector<int> tree;
//     // Depth of each element in tree
//     std::vector<int> depth;
    
    
//     public:
    
//     KDTree(Embedding <DIM> &embed) : embed(embed) {
        
//         // Set each element as null initially
//         tree.resize(embed.NV, -1);
        
//         // Set depth of each element to zero intialliy
//         depth.resize(embed.NV, 0);  
        
//         // Initial sorted list
//         std::vector<int> sorted(embed.NV);
//         std::iota(sorted.begin(), sorted.end(), 0);
        
//         py::print(sorted);
        
//         // Initial sort
//         auto cmp = [&embed](const int lhs, const int rhs) {
//             return embed.get_pos(lhs)(0) < embed.get_pos(rhs)(0);
//         };
//         std::sort(sorted.begin(), sorted.end(), cmp);
        
//         py::print(sorted);
        
//         // Set root
//         tree[embed.NV/2] = embed.NV/2;
        
//         Iterate through levels of tree starting at root
//         for(int h = 1; h < log2(embed.NV); h++) {
            
//             if(h > 2) {
//                 break;
//             }
            
//             py::print(embed.NV, "depth", h);
            
//             // Iterate through each pivot node
//             for(int bi = 1; bi < pow(2, h); bi += 2) {
                
//                 // Find pivot
//                 int pivot = embed.NV * bi / int(pow(2, h));
                
//                 // Set depth of pivot
//                 depth[pivot] = h-1;
                
//                 // Set children
//                 for(int bj = 1; bj < pow(2, h+1); bj += 2) {
//                     // Find pivot
//                     int child = embed.NV * bj / int(pow(2, h+1));
                    
//                     tree[child] = pivot;
//                 }
                
                
//             }
        
        
// //         // Iterate through levels of tree starting at root
// //         for(int h = 1; h < log2(embed.NV); h++) {
            
// //             if(h > 2) {
// //                 break;
// //             }
            
// //             py::print(embed.NV, "depth", h);
            
// //             // Iterate through each pivot node
// //             for(int b = 1; b < pow(2, h); b += 2) {
                
// //                 // Find pivot
// //                 int pivot = embed.NV * b / int(pow(2, h));
                
// //                 // Set depth of pivot
// //                 depth[pivot] = h-1;
                
// //                 // Find left and right sort boudaries
// //                 int left = embed.NV * (b-1) / int(pow(2, h));
// //                 int right = embed.NV * (b+1) / int(pow(2, h));            
                
// //                 py::print("pivot:", pivot, "left:", left, "right:", right);
                                
                
// //                 // Sort to either side of pivot
// //                 auto cmp = [&embed, h](const int lhs, const int rhs) {
// //                     return embed.get_pos(lhs)(h%DIM) < embed.get_pos(rhs)(h%DIM);
// //                 };
                
// //                 std::sort(sorted.begin()+left, sorted.begin()+pivot, cmp);
// //                 std::sort(sorted.begin()+pivot+1, sorted.begin()+right, cmp);
                
// //                 py::print(sorted);
                
                
// //             }
            
// //         }
        
//     }
    
//     std::vector<int> get_neighbors(int vi, double max_dist) {
        
//         return std::vector<int> (1, vi);

//     }
    
// };
    
    


    
#endif // SPACE_PART_HPP
