#ifndef VORONOI_HPP
#define VORONOI_HPP

#include "alpha_complex.hpp"

//Although everything is templated with "Dim", algorithms are only promised to
//work in 2d.


template <int DIM>
class Voronoi {

public:
  Embedding<DIM> embed;
  CellComplex comp;
  XVec voronoi_vertices;
  XVec cell_centroids;
  XVec cell_areas;
  std::vector<double> rad2;

  Voronoi();
  Voronoi(int NV, RXVec vert_pos, std::vector<double> rad2, RDMat box_mat, bool periodic);

  //construction subroutines
  void construct_voronoi_vertices();
  void construct_cell_areas_and_centroids();


  XVec get_cell_centroids() const;
  XVec get_cell_areas() const;
  Embedding<DIM> get_embedding() const;
};

template <int DIM>
Voronoi<DIM>::Voronoi() {}

template <int DIM>
Voronoi<DIM>::Voronoi(int NV, RXVec vert_pos, std::vector<double> _rad2, RDMat box_mat, bool periodic) :embed(NV, vert_pos, box_mat, periodic), comp(2), rad2(_rad2) {
  comp = construct_alpha_complex(embed,_rad2, false);
  construct_voronoi_vertices();
  construct_cell_areas_and_centroids();
}

template <int DIM>
void Voronoi<DIM>::construct_voronoi_vertices() {
  voronoi_vertices = XVec::Zero(DIM*comp.ndcells[2]);
  for (int v = 0; v < comp.ndcells[2]; v++) {
    auto neighb_cells = comp.get_faces(v+comp.dcell_range[2].first,0);
    std::vector<int> neighb_cells_vec(neighb_cells.begin(), neighb_cells.end());
    voronoi_vertices.segment<DIM>(DIM*v) = std::get<1>(calc_circumsphere(neighb_cells_vec,
        embed, rad2));
  }
}

//use the "shoelace formula" - need to compute vertices first.
//need to be careful with PBCs
template <int DIM>
void Voronoi<DIM>::construct_cell_areas_and_centroids() {
  cell_areas = XVec::Zero(comp.ndcells[0]);
  cell_centroids = XVec::Zero(DIM*comp.ndcells[0]);
  for (int c = 0; c < comp.ndcells[0]; c++) {
    auto neighb_vertices = comp.get_cofaces(c+comp.dcell_range[0].first, 2); //this is actually an unordered set
    auto neighb_edges = comp.get_cofaces(c+comp.dcell_range[0].first, 1);
    int n = neighb_vertices.size();
    //here we have an algorithm to sort the vertices.
    std::vector<int> sorted_vertices(n);
    sorted_vertices[0] = *neighb_vertices.begin();//beginning of cycle is arbitrary (as is order, as long as we abs(area) at the end)
    std::vector<DVec> unrolled_pos(n); 
    unrolled_pos[0] = voronoi_vertices.segment<DIM>(DIM*(sorted_vertices[0]-comp.dcell_range[2].first));
    std::unordered_set<int> seen;
  
    for(int i = 0; i < n-1; i++) {

      auto edges = comp.get_faces(sorted_vertices[i],1);
      //take the first(i.e. arbitrary) element of the complement of seen, in the intersection of neighb_edges and edges.
      std::vector<int> intersection;
      std::copy_if(edges.begin(), edges.end(), std::back_inserter(intersection), [&neighb_edges] (int i) {return neighb_edges.find(i) != neighb_edges.end();});
      std::vector<int> complement; 
      std::copy_if(intersection.begin(), intersection.end(), std::back_inserter(complement), [&seen] (int i) {return seen.find(i) == seen.end();});
      int edge = complement[0]; 
      seen.insert(edge); 
  
      //now find the right point from edge and put it in sorted vertices
      auto points = comp.get_cofaces(edge, 2); 
      if(*points.begin() == sorted_vertices[i]) {sorted_vertices[i+1] = *std::next(points.begin());}
      else { assert(*std::next(points.begin()) == sorted_vertices[i]); sorted_vertices[i+1] = *points.begin(); } 
      
      unrolled_pos[i+1] = unrolled_pos[i] + embed.get_diff_realpos(voronoi_vertices.segment<DIM>(DIM*(sorted_vertices[i]-comp.dcell_range[2].first)), voronoi_vertices.segment<DIM>(DIM*(sorted_vertices[i+1]-comp.dcell_range[2].first))); 
    }

    for (int i = 0; i < n ; i++) {
      cell_areas[c] += 0.5*(unrolled_pos[i](0)*unrolled_pos[(i+1) % n](1)-unrolled_pos[(i+1) % n](0)*unrolled_pos[i](1));
    }
    for (int i = 0; i < n; i++) {
      double temp = ((1.0)/(6.0*cell_areas[c]))*(unrolled_pos[i](0)*unrolled_pos[(i+1) % n](1)-unrolled_pos[(i+1) % n](0)*unrolled_pos[i](1));
      cell_centroids.segment<DIM>(DIM*c) += temp*(unrolled_pos[i] + unrolled_pos[(i+1)%n]);
    }
   
    cell_areas[c] = abs(cell_areas[c]); 
    //push centroid in bounds
    DMat inv = embed.box_mat.inverse(); 
    DVec vpos = inv*cell_centroids.segment<DIM>(DIM*c); 
    vpos[0] -= std::floor(vpos[0]);
    vpos[1] -= std::floor(vpos[1]);
    cell_centroids.segment<DIM>(DIM*c) = embed.box_mat*vpos;

  }
}

template <int DIM>
XVec Voronoi<DIM>::get_cell_centroids() const {
  return cell_centroids;
}

template <int DIM>
XVec Voronoi<DIM>::get_cell_areas() const {
  return cell_areas;
}

template<int DIM>
Embedding<DIM> Voronoi<DIM>::get_embedding() const {
  return embed;
}

#endif
