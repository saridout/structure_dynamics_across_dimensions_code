#ifndef GRAPHCOMPLEX_HPP
#define GRAPHCOMPLEX_HPP
    
#include "eigen_macros.hpp"
#include "cell_complex.hpp"
#include "embedding.hpp"

#include <pybind11/pybind11.h>    
namespace py = pybind11;

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>


    
class Graph {
    
public:

    // Number of vertices
    int NV;

    // Number of edges
    int NE;
    // Nodes for each edge
    std::vector<int> edgei;
    std::vector<int> edgej;

    bool has_list;
    std::vector<std::unordered_set<int> > neighbors; 
    
    Graph(int NV, int NE, std::vector<int> &edgei, std::vector<int> &edgej) :
         NV(NV), NE(NE), edgei(edgei), edgej(edgej) {
             has_list = false;
         };

    void construct_neighbor_list() {

        if(has_list) {
            return;
        } else {
            has_list = true;
        }

        neighbors.resize(NV);

        for(int ei = 0; ei < NE; ei++) {
            int vi = edgei[ei];
            int vj = edgej[ei];

            neighbors[vi].insert(vj);
            neighbors[vj].insert(vi);

        }
    }
};

    
CellComplex construct_graph_complex(Graph &graph, bool oriented) {
        
    CellComplex comp(1, true, oriented);
       
    // Vertices are guarenteed to have matching cell number and label
    for(int i = 0; i < graph.NV; i++) {
        std::vector<int> facets;
        std::vector<int> coeffs;
        comp.add_cell(i, 0, facets, coeffs);
    }


    for(int i = 0; i < graph.NE; i++) {
        std::vector<int> facets;
        facets.push_back(graph.edgei[i]);
        facets.push_back(graph.edgej[i]);

        std::vector<int> coeffs;
        if(oriented) {
            coeffs.push_back(-1);
            coeffs.push_back(1);
        }
        comp.add_cell(i, 1, facets, coeffs);
 
    }
    
    
    comp.construct_cofacets();
    comp.make_compressed(); 

    return comp;
    
}



template <int DIM> XVec calc_edge_extensions(RXVec disp, Graph &graph, Embedding<DIM> &embed, bool is_strain=false) {
        
    XVec ext = XVec::Zero(graph.NE);
    
    for(int i = 0; i < graph.NE; i++) {

        int vi = graph.edgei[i];
        int vj = graph.edgej[i];
        
        DVec posi = embed.get_vpos(vi);
        DVec posj = embed.get_vpos(vj);
        
        DVec bvec = posj - posi;
        
        for(int d = 0; d < DIM; d++) {
            if(std::fabs(bvec(d)) > 0.5) {
                bvec(d) -= ((bvec(d) > 0) - (bvec(d) < 0));
            }
        }
        
        bvec = embed.box_mat * bvec;
                
        double l0 = bvec.norm();
        
        bvec.normalize();
        
        DVec ui = disp.segment<DIM>(DIM*vi);
        DVec uj = disp.segment<DIM>(DIM*vj);
        
        DVec du = uj - ui;
        
        double eps = bvec.dot(du);
        
        if(is_strain) {
            eps /= l0;
        }
        
        ext(i) = eps;
            
        
    }
    
    return ext;
    
}


template <int DIM> std::tuple<int, XVec, int, XiVec, XiVec, DVec> 
    convert_to_network(CellComplex &comp, Embedding<DIM> &embed) {
    
    int NE = comp.ndcells[1];
    XiVec edgei = XiVec::Zero(NE);
    XiVec edgej = XiVec::Zero(NE);
    
    for(int c = comp.dcell_range[1].first; c < comp.dcell_range[1].second; c++) {
        auto facets = comp.get_facets(c);
        
        edgei[c-comp.dcell_range[1].first] = facets[0];
        edgej[c-comp.dcell_range[1].first] = facets[1];
        
    }
    
    XVec node_pos = XVec::Zero(DIM*embed.NV);
    
    for(int i = 0; i < embed.NV; i++) {
        node_pos.segment<DIM>(DIM*i) = embed.get_pos(i);
    }
    
    return std::make_tuple(embed.NV, node_pos, NE, edgei, edgej, embed.box_mat.diagonal());
    
}


    
#endif // GRAPHCOMPLEX_HPP