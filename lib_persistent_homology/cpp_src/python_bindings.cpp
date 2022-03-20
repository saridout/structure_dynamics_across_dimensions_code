#include "embedding.hpp"
#include "space_partition.hpp"
#include "filtration.hpp"

#include "cell_complex.hpp"
#include "graph_complex.hpp"
// #include "corner_complex.hpp"
#include "cubical_complex.hpp"
#include "morse_complex.hpp"
#include "extended_complex.hpp"

#include "deformation.hpp"


#include "search.hpp"
#include "persistent_homology.hpp"
#include "persistence_simplification.hpp"
#include "persistence_landscape.hpp"

#ifdef ALPHA
    #include "alpha_complex.hpp"
#endif

#ifdef OPTIMAL
    #include "optimal.hpp"
#endif

#include "protein_algs.hpp"
#include "softness.hpp"

#include "voronoi.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>


namespace py = pybind11;

template <int DIM> void init_embedding_templates(py::module &m) {

    py::class_<Embedding<DIM> >(m, (std::string("Embedding")+std::to_string(DIM)+std::string("D")).c_str())
        .def_readonly("dim", &Embedding<DIM>::dim)
        .def_readonly("NV", &Embedding<DIM>::NV)
        .def_readonly("box_mat", &Embedding<DIM>::box_mat)
        .def_readonly("box_mat_inv", &Embedding<DIM>::box_mat_inv)
        .def_readonly("periodic", &Embedding<DIM>::periodic)
        .def(py::init<int, RXVec, RDMat, bool>())
        .def("get_vpos", &Embedding<DIM>::get_vpos)
        .def("get_pos", &Embedding<DIM>::get_pos)
        .def("transform", &Embedding<DIM>::transform)
        .def("get_vdiff", (DVec (Embedding<DIM>::*)(int, int)) &Embedding<DIM>::get_vdiff)
        .def("get_vdiff", (DVec (Embedding<DIM>::*)(DVec const &, DVec const &)) &Embedding<DIM>::get_vdiff)
        .def("get_diff", (DVec (Embedding<DIM>::*)(int, int)) &Embedding<DIM>::get_diff)
        .def("get_diff", (DVec (Embedding<DIM>::*)(DVec const &, DVec const &)) &Embedding<DIM>::get_diff);
    
    py::class_<SpacePartition<DIM>>(m, (std::string("SpacePartition")+std::to_string(DIM)+std::string("D")).c_str())
        .def("get_neighbors", &SpacePartition<DIM>::get_neighbors);
    
    py::class_<NeighborGrid<DIM>, SpacePartition<DIM> >(m, (std::string("NeighborGrid")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<Embedding<DIM>&, double>());
    
//     py::class_<KDTree<DIM>, SpacePartition<DIM> >(m, (std::string("KDTree")+std::to_string(DIM)+std::string("D")).c_str())
//         .def(py::init<Embedding<DIM>& >());


}

template <int DIM> void init_graph_templates(py::module &m) {

    m.def((std::string("calc_edge_extensions_")+std::to_string(DIM)+std::string("D")).c_str(), &calc_edge_extensions<DIM>,
         py::arg("disp"), py::arg("graph"), py::arg("embed"), py::arg("is_strain") = false);

    m.def((std::string("convert_to_network_")+std::to_string(DIM)+std::string("D")).c_str(), &convert_to_network<DIM>);

}

// template <int DIM> void init_corner_templates(py::module &m) {

//     m.def((std::string("construct_corner_complex_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &construct_corner_complex<DIM>);

//     m.def((std::string("find_corners_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &find_corners<DIM>);
//     m.def((std::string("calc_corner_strains_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &calc_corner_strains<DIM>, py::arg("corners"), py::arg("disp"), py::arg("embed"), py::arg("strain")=false);

//     m.def((std::string("calc_corner_flatness_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &calc_corner_flatness<DIM>);


// }

#ifdef ALPHA
template <int DIM> void init_alpha_templates(py::module &m) {

    m.def((std::string("calc_circumsphere_")+std::to_string(DIM)+std::string("D")).c_str(), &calc_circumsphere<DIM>);

    m.def((std::string("construct_alpha_complex_")+std::to_string(DIM)+std::string("D")).c_str(), &construct_alpha_complex<DIM>,
         py::arg("embed"), py::arg("weights"), py::arg("oriented")=false, py::arg("dim_cap")=DIM);

    m.def((std::string("calc_alpha_vals_")+std::to_string(DIM)+std::string("D")).c_str(), &calc_alpha_vals<DIM>,
         py::arg("comp"), py::arg("embed"), py::arg("weights"), py::arg("alpha0")=-1.0);


    m.def((std::string("join_dtriangles_")+std::to_string(DIM)+std::string("D")).c_str(), &join_dtriangles<DIM>,
        py::arg("comp"), py::arg("alpha_vals"), py::arg("threshold")=0.0);

     m.def(("calc_radial_gap_distribution_"+std::to_string(DIM)+"D").c_str(), &calc_radial_gap_distribution<DIM>,
          py::arg("cell_list"), py::arg("comp"), py::arg("embed"), py::arg("rad"),  py::arg("max_dist")=-1, py::arg("verbose")=false);


    m.def(("calc_gap_angle_class"+std::to_string(DIM)+std::string("D")).c_str(), &calc_gap_angle_class<DIM>, py::arg("cell_list"),py::arg("alpha_vals"),py::arg("comp"),py::arg("Embedding"),py::arg("num_angles")=128,py::arg("verbose")=false);

}

template <int DIM> void init_voronoi_templates(py::module &m) {

    py::class_<Voronoi<DIM> >(m, (std::string("Voronoi")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<int, RXVec, std::vector<double> , RDMat, bool>())
        .def_readonly("comp", &Voronoi<DIM>::comp)
        .def_readonly("embed", &Voronoi<DIM>::embed)
        .def("get_cell_centroids", &Voronoi<DIM>::get_cell_centroids)
        .def("get_cell_areas", &Voronoi<DIM>::get_cell_areas)
        .def("get_embedding", &Voronoi<DIM>::get_embedding);


}

#endif


template <int DIM> void init_deform_templates(py::module &m) {


    m.def((std::string("calc_def_grad_")+std::to_string(DIM)+std::string("D")).c_str(), 
          (std::tuple<DMat, double> (*) (std::vector<int>&, RXVec, Embedding<DIM>&, RXVec, bool)) &calc_def_grad<DIM>,
         py::arg("verts"), py::arg("disp"), py::arg("embed"), py::arg("weights"), py::arg("calc_D2min")=false);
    
    m.def((std::string("calc_def_grad_")+std::to_string(DIM)+std::string("D")).c_str(), 
          (std::tuple<DMat, double> (*) (std::vector<int>&, RXVec, Embedding<DIM>&, bool)) &calc_def_grad<DIM>,
         py::arg("verts"), py::arg("disp"), py::arg("embed"), py::arg("calc_D2min")=false);

    m.def((std::string("def_grad_to_strain_")+std::to_string(DIM)+std::string("D")).c_str(), &def_grad_to_strain<DIM>,
         py::arg("F"), py::arg("linear")=true);
    
    m.def((std::string("decompose_def_grad_")+std::to_string(DIM)+std::string("D")).c_str(), &decompose_def_grad<DIM>,
         py::arg("F"), py::arg("linear")=true);

    m.def((std::string("subtract_global_motion_")+std::to_string(DIM)+std::string("D")).c_str(), &subtract_global_motion<DIM>,
         py::arg("disp"), py::arg("embed"), py::arg("linear")=true);


    m.def((std::string("calc_local_rmsd_")+std::to_string(DIM)+std::string("D")).c_str(),
      &calc_local_rmsd<DIM>, py::arg("disp"), py::arg("embed"), py::arg("part"), py::arg("max_dist"), py::arg("linear")=true, py::arg("weighted")=false);
        
    
    m.def((std::string("calc_tri_strains_")+std::to_string(DIM)+std::string("D")).c_str(), &calc_tri_strains<DIM>,
         py::arg("disp"), py::arg("comp"), py::arg("embed"), py::arg("linear")=true);

    m.def((std::string("calc_delaunay_D2min_strain_")+std::to_string(DIM)+std::string("D")).c_str(),
          &calc_delaunay_D2min_strain<DIM>,
         py::arg("disp"),  py::arg("comp"), py::arg("embed"), py::arg("max_dist") = 2, py::arg("linear")=true);


    m.def((std::string("calc_delaunay_local_rmsd_")+std::to_string(DIM)+std::string("D")).c_str(),
          &calc_delaunay_local_rmsd<DIM>,
         py::arg("disp"),  py::arg("comp"), py::arg("embed"), py::arg("max_dist") = 2, py::arg("linear")=true);

    m.def((std::string("calc_grouped_delaunay_D2min_strain_")+std::to_string(DIM)+std::string("D")).c_str(),
          &calc_grouped_delaunay_D2min_strain<DIM>,
         py::arg("groups"), py::arg("disp"),  py::arg("comp"), py::arg("embed"), py::arg("max_dist") = 2, py::arg("linear")=true);


//     m.def((std::string("calc_stresses_")+std::to_string(DIM)+std::string("D")).c_str(), &calc_stresses<DIM>);

//     m.def((std::string("calc_voronoi_D2min_")+std::to_string(DIM)+std::string("D")).c_str(),
//           (XVec (*) (RXVec, CellComplex&, Embedding<DIM>&, int)) &calc_voronoi_D2min<DIM>,
//          py::arg("disp"),  py::arg("comp"), py::arg("embed"), py::arg("max_dist") = 2);



    m.def((std::string("calc_voronoi_D2min_")+std::to_string(DIM)+std::string("D")).c_str(), &calc_voronoi_D2min<DIM>);

    m.def((std::string("calc_flatness_")+std::to_string(DIM)+std::string("D")).c_str(), &calc_flatness<DIM>);


}


template <int DIM> void init_search_templates(py::module &m) {

    m.def((std::string("calc_euclid_pair_dists_")+std::to_string(DIM)+std::string("D")).c_str(),
          &calc_euclid_pair_dists<DIM>);

    m.def((std::string("calc_euclid_point_dists_")+std::to_string(DIM)+std::string("D")).c_str(),
          &calc_euclid_point_dists<DIM>);

    m.def((std::string("get_point_neighborhood_")+std::to_string(DIM)+std::string("D")).c_str(),
          &get_point_neighborhood<DIM>);



}


template <int DIM> void init_protein_templates(py::module &m) {

//     m.def((std::string("shrink_alpha_complex_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &shrink_alpha_complex<DIM>, py::arg("comp"), py::arg("filt"), py::arg("max_dist"), py::arg("threshold")=0.0, py::arg("verbose")=false);

    m.def((std::string("shrink_alpha_complex_")+std::to_string(DIM)+std::string("D")).c_str(),
          &shrink_alpha_complex<DIM>);



//     m.def((std::string("get_contact_network_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &get_contact_network<DIM>);

    m.def((std::string("transform_coords_")+std::to_string(DIM)+std::string("D")).c_str(),
          &transform_coords<DIM>);
    
    m.def((std::string("calc_com_")+std::to_string(DIM)+std::string("D")).c_str(),
          &calc_com<DIM>);
    
    m.def((std::string("calc_hinge_overlap_")+std::to_string(DIM)+std::string("D")).c_str(),
          &calc_hinge_overlap<DIM>,
         py::arg("sector"), py::arg("disp"), py::arg("embed"), py::arg("linear")=true);
    
    
//     m.def((std::string("calc_rmsd_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &calc_rmsd<DIM>);
    
//     m.def((std::string("calc_rmsd_")+std::to_string(DIM)+std::string("D")).c_str(),
//       &calc_rmsd<DIM>, py::arg("verts"), py::arg("disp"), py::arg("embed"), py::arg("linear")=true);

   
//     m.def((std::string("calc_rmsd_err_")+std::to_string(DIM)+std::string("D")).c_str(),
//       &calc_rmsd_err<DIM>, py::arg("verts"), py::arg("disp"), py::arg("sigma"), py::arg("n_iters"), py::arg("embed"), py::arg("max_dist"), py::arg("linear")=true);
    
    m.def((std::string("calc_lrmsd_err_")+std::to_string(DIM)+std::string("D")).c_str(),
      &calc_lrmsd_err<DIM>, py::arg("vi"), py::arg("disp"), py::arg("embed"), py::arg("part"), py::arg("max_dist"), 
          py::arg("sigma_ref"), py::arg("sigma_def"), py::arg("n_iters"), py::arg("linear")=true, py::arg("weighted")=true);
    
    m.def((std::string("calc_hinge_overlap_err_")+std::to_string(DIM)+std::string("D")).c_str(),
      &calc_hinge_overlap_err<DIM>, py::arg("sector"), 
          py::arg("disp"), py::arg("embed"), py::arg("sigma_ref"), py::arg("sigma_def"), py::arg("n_iters"), py::arg("linear")=true);
    
//     m.def((std::string("calc_local_strain_")+std::to_string(DIM)+std::string("D")).c_str(),
//       &calc_local_strain<DIM>, py::arg("disp"), py::arg("embed"), py::arg("max_dist"), py::arg("linear")=true);
    
//     m.def((std::string("calc_local_strain_order2_")+std::to_string(DIM)+std::string("D")).c_str(),
//       &calc_local_strain_order2<DIM>, py::arg("disp"), py::arg("embed"), py::arg("max_dist"), py::arg("linear")=true);
    
//     m.def((std::string("calc_basin_boundary_dists_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &calc_basin_boundary_dists<DIM>);
    
//     m.def((std::string("calc_basin_com_dists_")+std::to_string(DIM)+std::string("D")).c_str(),
//           &calc_basin_com_dists<DIM>);


}


PYBIND11_MODULE(phom, m) {

    // Cell complex

    py::class_<CellComplex>(m, "CellComplex")
        .def_readonly("dim", &CellComplex::dim)
        .def_readonly("ncells", &CellComplex::ncells)
        .def_readonly("ndcells", &CellComplex::ndcells)
        .def_readonly("dcell_range", &CellComplex::dcell_range)
        .def_readonly("regular", &CellComplex::regular)
        .def_readonly("oriented", &CellComplex::oriented)
        .def(py::init<int, bool, bool>(), py::arg("dim"), py::arg("regular")=true, py::arg("oriented")=false)
       . def(py::init<CellComplex>())
        .def("add_cell", (void (CellComplex::*)(int, int, std::vector<int>&, std::vector<int>&)) &CellComplex::add_cell)
        .def("get_dim", &CellComplex::get_dim)
        .def("get_label", &CellComplex::get_label)
        .def("get_facets", &CellComplex::get_facets)
        .def("get_cofacets", &CellComplex::get_cofacets)
        .def("get_coeffs", &CellComplex::get_coeffs)
        .def("get_faces", &CellComplex::get_faces, py::arg("alpha"), py::arg("target_dim")=-1)
        .def("get_cofaces", &CellComplex::get_cofaces, py::arg("alpha"), py::arg("target_dim")=-1)
        .def("get_star", &CellComplex::get_star, py::arg("alpha"), py::arg("dual"), py::arg("target_dim")=-1)
        .def("get_labels", &CellComplex::get_labels)
        .def("make_compressed", &CellComplex::make_compressed)
        .def("construct_cofacets", &CellComplex::construct_cofacets);


    m.def("prune_cell_complex", &prune_cell_complex);
    m.def("prune_cell_complex_map", &prune_cell_complex_map);
    m.def("prune_cell_complex_sequential", &prune_cell_complex_sequential,
          py::arg("comp"), py::arg("priority"), py::arg("preserve"), py::arg("preserve_stop")=true, py::arg("allow_holes")=false, py::arg("threshold")=0.0, py::arg("target_dim")=-1);
    m.def("prune_cell_complex_sequential_surface", &prune_cell_complex_sequential_surface,
          py::arg("comp"), py::arg("priority"), py::arg("preserve"), py::arg("surface"), py::arg("preserve_stop")=true, py::arg("allow_holes")=false, py::arg("threshold")=0.0, py::arg("target_dim")=-1, py::arg("verbose")=false);
    m.def("check_boundary_op", &check_boundary_op,
          "Checks the boundary operator of a complex to ensure that \\delta_d\\delta_(d-1) = 0 for each cell.");
//         m.def("get_boundary", &get_boundary);


    // Filtration

    py::class_<Filtration>(m, "Filtration")
        .def_readonly("ncells", &Filtration::ncells)
        .def_readonly("ascend", &Filtration::ascend)
        .def(py::init<CellComplex&, RXVec, RXiVec, RXiVec, bool, int>(),
             py::arg("comp"), py::arg("func"), py::arg("digi_func"),
             py::arg("order"), py::arg("ascend")=true, py::arg("filt_dim")=-1)
        .def("get_func", &Filtration::get_func)
        .def("get_digi_func", &Filtration::get_digi_func)
        .def("get_order", &Filtration::get_order)
        .def("get_filtration", &Filtration::get_filtration);


    m.def("construct_filtration", &construct_filtration,
         py::arg("comp"), py::arg("func"), py::arg("ascend")=true);
    m.def("construct_induced_filtration", &construct_induced_filtration,
         py::arg("comp"), py::arg("func"), py::arg("func_order"),py::arg("filt_dim"), py::arg("ascend")=true);
    m.def("reduce_filtration", &reduce_filtration);
    //     m.def("perform_watershed_transform", &perform_watershed_transform,
    //           py::arg("time"), py::arg("comp"), py::arg("ascend")=true, py::arg("co")=false);
    //


    // Embedding
    init_embedding_templates<1>(m);
    init_embedding_templates<2>(m);
    init_embedding_templates<3>(m);
    init_embedding_templates<4>(m);
    init_embedding_templates<5>(m);

    // Graph complex

    init_graph_templates<1>(m);
    init_graph_templates<2>(m);

    py::class_<Graph>(m, "Graph")
        .def(py::init<int, int, std::vector<int>&, std::vector<int>&>())
        .def_readonly("NV", &Graph::NV)
        .def_readonly("NE", &Graph::NE)
        .def_readonly("edgei", &Graph::edgei)
        .def_readonly("edgej", &Graph::edgej)
        .def_readonly("neighbors", &Graph::neighbors)
        .def("construct_neighbor_list", &Graph::construct_neighbor_list);

    m.def("construct_graph_complex", &construct_graph_complex);

    // Corner complex

//     init_corner_templates<1>(m);
//     init_corner_templates<2>(m);
///     init_corner_templates<3>(m);
//     init_corner_templates<4>(m);
//     init_corner_templates<5>(m);
    // Cubical complex

    m.def("construct_cubical_complex", &construct_cubical_complex);
    m.def("construct_masked_cubical_complex", &construct_masked_cubical_complex);
    m.def("get_boundary_pixels", &get_boundary_pixels);
    m.def("calc_elongation", &calc_elongation);
    m.def("construct_hypercube_complex", &construct_hypercube_complex, py::arg("dim"), py::arg("verbose")=false);

#ifdef ALPHA

    // Alpha complex

    init_alpha_templates<2>(m);
    init_alpha_templates<3>(m);
    init_alpha_templates<4>(m);
    init_alpha_templates<5>(m);

    m.def("calc_radial_edge_counts", &calc_radial_edge_counts,
          py::arg("cell_list"), py::arg("edge_types"), py::arg("comp"), py::arg("max_dist")=-1);


    m.def("calc_radial_tri_distribution", &calc_radial_tri_distribution,
          py::arg("cell_list"), py::arg("alpha_vals"), py::arg("comp"), py::arg("max_dist")=-1, py::arg("verbose")=false);

    m.def("calc_angular_gap_distribution", &calc_angular_gap_distribution,
          py::arg("cell_list"), py::arg("alpha_vals"), py::arg("comp"), py::arg("max_dist")=-1, py::arg("verbose")=false);


    m.def("calc_radial_cycle_distribution", &calc_radial_cycle_distribution,
          py::arg("cell_list"), py::arg("alpha_vals"), py::arg("comp"), py::arg("vertex_types"), py::arg("max_dist")=-1, py::arg("verbose")=false);
    
    
    // Voronoi complex
    
//     init_voronoi_templates<2>(m);

    

#endif

    // Deformation calculations
    init_deform_templates<2>(m);
    init_deform_templates<3>(m);
    init_deform_templates<4>(m);
    init_deform_templates<5>(m);
    // Morse complex

    m.def("get_star_decomp", &get_star_decomp,
          py::arg("alpha"), py::arg("filt"), py::arg("comp"), py::arg("target_dim")=-1);
    m.def("construct_discrete_gradient", &construct_discrete_gradient);
    m.def("traverse_flow", &traverse_flow, py::arg("s"), py::arg("V"),
          py::arg("comp"), py::arg("co")=false, py::arg("coordinated")=false);
    m.def("find_morse_boundary", &find_morse_boundary, py::arg("s"), py::arg("V"),
          py::arg("comp"), py::arg("co")=true, py::arg("oriented")=false);
    m.def("construct_morse_complex", &construct_morse_complex, py::arg("V"),
          py::arg("comp"),  py::arg("oriented")=false);
    m.def("construct_morse_filtration", &construct_morse_filtration);
    m.def("find_connections", &find_connections);

    // Morse feature extraction

    m.def("find_morse_feature", &find_morse_feature,
         py::arg("s"), py::arg("V"), py::arg("comp"), py::arg("co")=false);
    m.def("find_morse_features", &find_morse_features,
         py::arg("cells"), py::arg("V"), py::arg("comp"), py::arg("co")=false);
    m.def("find_morse_cells", &find_morse_cells,
         py::arg("cells"), py::arg("V"), py::arg("mcomp"), py::arg("comp"), py::arg("co")=false);

    m.def("find_morse_basins", &find_morse_basins);
    m.def("find_morse_skeleton", &find_morse_skeleton,
         py::arg("V"), py::arg("morse_comp"), py::arg("comp"), py::arg("skeleton_dim")=1);

    m.def("convert_feature_dim", &convert_feature_dim,
          py::arg("feature"), py::arg("target_dim"), py::arg("filt"), py::arg("comp"), py::arg("inclusive")=true);

    m.def("convert_morse_voids_to_basins", &convert_morse_voids_to_basins);


    //     m.def("find_morse_basin_borders", &find_morse_basin_borders,
    //          py::arg("mcomp"), py::arg("V"), py::arg("coV"), py::arg("filt"), py::arg("comp"), py::arg("target_dim")=-1);
    //     m.def("extract_morse_basin", &extract_morse_basin,
    //          py::arg("s"), py::arg("mcomp"), py::arg("V"), py::arg("coV"),
    //           py::arg("comp"), py::arg("filt"), py::arg("target_dim")=-1);
    //     m.def("extract_morse_feature", &extract_morse_feature,
    //          py::arg("i"), py::arg("j"), py::arg("mcomp"), py::arg("filt"), py::arg("target_dim")=-1,
    //     py::arg("complement")=false);
    //     m.def("extract_morse_feature_to_real", &extract_morse_feature_to_real,
    //          py::arg("i"), py::arg("j"), py::arg("mcomp"), py::arg("V"), py::arg("coV"),
    //           py::arg("comp"), py::arg("filt"), py::arg("complement")=false, py::arg("target_dim")=-1);


    // Extended complex

    m.def("extend_complex", &extend_complex);
    m.def("extend_filtration", &extend_filtration);
    m.def("extend_discrete_gradient", &extend_discrete_gradient);
    m.def("find_morse_smale_basins", &find_morse_smale_basins);
    m.def("find_morse_smale_skeleton", &find_morse_smale_skeleton,
         py::arg("pairs"), py::arg("V"), py::arg("morse_comp"), py::arg("ext_comp"), py::arg("skeleton_dim")=1);

    // Cell complex searching


    init_search_templates<2>(m);
    init_search_templates<3>(m);
    init_search_templates<4>(m);
    init_search_templates<5>(m);

    m.def("perform_bfs", &perform_bfs);
    m.def("calc_comp_pair_dists", &calc_comp_pair_dists);
    m.def("calc_comp_point_dists", &calc_comp_point_dists,
         py::arg("p"), py::arg("comp"), py::arg("max_dist")=-1);
    m.def("find_nearest_neighbors", &find_nearest_neighbors,
         py::arg("p"), py::arg("comp"), py::arg("max_dist"), py::arg("target_dim")=-1);
    m.def("find_neighbors", &find_neighbors,
         py::arg("p"), py::arg("comp"), py::arg("max_dist"), py::arg("target_dim")=-1);
    m.def("calc_cell_pair_dist", &calc_cell_pair_dist);
    m.def("find_local_extrema",
          (std::tuple<std::vector<int>, std::vector<int> > (*) (RXVec, CellComplex&, int)) &find_local_extrema,
         py::arg("heigh"), py::arg("comp"), py::arg("max_dist")=1);
    m.def("find_local_extrema",
          (std::tuple<std::vector<int>, std::vector<int> > (*) (RXVec, CellComplex&, std::vector<bool>&)) &find_local_extrema);
    //     m.def("find_thresholded_component", &find_thresholded_component);



    // Persistent homology

    m.def("calc_persistence", &calc_persistence);
    m.def("calc_betti_numbers", &calc_betti_numbers);

    m.def("calc_extended_persistence", (std::tuple<std::tuple<std::vector<std::pair<int, int> >,
                                        std::vector<std::pair<int, int> >,
                                        std::vector<std::pair<int, int> > >,
                                        std::unordered_map<int, std::vector<int> > > (*)
                                        (Filtration&, Filtration&, CellComplex&, bool, int)) &calc_extended_persistence,
                                     py::arg("filt_asc"), py::arg("filt_desc"), py::arg("comp"),
                                     py::arg("ext_cycles"), py::arg("dim")=-1);

    m.def("calc_extended_persistence", (std::tuple<std::vector<std::pair<int, int> >,
                                        std::vector<std::pair<int, int> >,
                                        std::vector<std::pair<int, int> > > (*)
                                        (Filtration&, Filtration&, CellComplex&)) &calc_extended_persistence,
                                     py::arg("filt_asc"), py::arg("filt_desc"), py::arg("comp"));


    m.def("calc_birth_cycles", &calc_birth_cycles, py::arg("filt"), py::arg("comp"), py::arg("dim")=-1);
    m.def("calc_homologous_birth_cycles", &calc_homologous_birth_cycles, py::arg("filt"), py::arg("comp"), py::arg("dim")=-1);

    m.def("extract_persistence_feature", &extract_persistence_feature,
         py::arg("i"), py::arg("j"), py::arg("comp"), py::arg("filt"), py::arg("target_dim")=-1, py::arg("complement")=false);


#ifdef OPTIMAL

    // Optimal cycles

    m.def("calc_optimal_cycles", &calc_optimal_cycles,
          py::arg("filt"), py::arg("comp"), py::arg("weights"), py::arg("dim")=-1, py::arg("verbose")=false,
         py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>());

    //     m.def("calc_optimal_homologous_cycles", &calc_optimal_homologous_cycles,
    //           py::arg("filt"), py::arg("comp"), py::arg("weights"), py::arg("dim")=-1, py::arg("verbose")=false,
    //          py::call_guard<py::scoped_ostream_redirect,
    //                      py::scoped_estream_redirect>());
#endif

    // Persistence simplification

    // need to type these
//     m.def("simplify_morse_complex",
//           (void (*)(double, RXiVec, RXiVec, Filtration&, CellComplex&, bool, bool)) &simplify_morse_complex,
//           py::arg("threshold"), py::arg("V"), py::arg("coV"), py::arg("comp"),
//           py::arg("filt"), py::arg("leq") = true, py::arg("verbose") = false);
//     m.def("simplify_morse_complex",
//           (void (*)(std::unordered_set<int>&, RXiVec, RXiVec, Filtration&, CellComplex&, bool, bool))
//           &simplify_morse_complex,
//           py::arg("preserve_pairs"), py::arg("V"), py::arg("coV"), py::arg("comp"),
//           py::arg("filt"), py::arg("finagle")=false, py::arg("verbose") = false);
    m.def("simplify_morse_complex",
          (void (*)(std::pair<int,int>, RXiVec, RXiVec, Filtration&, CellComplex&, int, bool, bool))
          &simplify_morse_complex,
          py::arg("pair"), py::arg("V"), py::arg("coV"), py::arg("filt"),
          py::arg("comp"), py::arg("target_dim")=-1, py::arg("cancel_target_pair")=false, py::arg("verbose") = false);

    m.def("simplify_morse_complex",
          (void (*)(double, RXiVec, RXiVec, Filtration&, CellComplex&, int, bool, bool))
          &simplify_morse_complex,
          py::arg("threshold"), py::arg("V"), py::arg("coV"), py::arg("filt"),
          py::arg("comp"), py::arg("target_dim")=-1, py::arg("parallel")=false, py::arg("verbose") = false);


    m.def("find_cancel_order", &find_cancel_order,
          py::arg("V"), py::arg("coV"), py::arg("filt"),
          py::arg("comp"), py::arg("target_dim")=-1, py::arg("parallel")=false, py::arg("verbose") = false);


    m.def("find_join_pair", &find_join_pair,
         py::arg("cells"), py::arg("V"), py::arg("coV"), py::arg("filt"), py::arg("comp"),
         py::arg("ntarget_cells")=1, py::arg("verbose")=false);

//     m.def("find_cancel_threshold", &find_cancel_threshold);
//     m.def("find_cancel_order", &find_cancel_order);
//     m.def("find_join_threshold", &find_join_threshold);
//     m.def("find_join_feature",
//           (std::tuple<double, std::pair<int, int>> (*)(std::vector<int>&, RXiVec, RXiVec, Filtration&, CellComplex&, int, bool)) &find_join_feature);
//     m.def("find_join_feature",
//           (std::tuple<double, std::pair<int, int>> (*)(std::vector<int>&, std::vector<int>&, RXiVec, RXiVec, Filtration&, CellComplex&, bool))  &find_join_feature);


//



//     m.def("extract_persistence_feature", &extract_persistence_feature,
//          py::arg("i"), py::arg("j"), py::arg("comp"), py::arg("filt"), py::arg("target_dim")=-1, py::arg("complement")=false);

//     // m.def("calc_persistence_landscape", (std::vector<std::vector<double> > (*)(std::vector<double>&, std::vector<double>&, std::vector<double>&, int)) &calc_persistence_landscape);
//     // m.def("calc_persistence_landscape", (std::vector<std::vector<double> > (*)(std::vector<std::pair<int, int> >&, std::vector<double>&, int, StarFiltration&)) &calc_persistence_landscape);



    m.def("construct_landscape", &construct_landscape);
    m.def("eval_landscape", &eval_landscape);
    m.def("combine_landscapes",
          (std::vector< std::pair<std::vector<double>, std::vector<double> > >
           (*)(std::vector< std::pair<std::vector<double>, std::vector<double> > >&,
            std::vector< std::pair<std::vector<double>, std::vector<double> > >&, double, double)) &combine_landscapes);

    m.def("combine_landscapes",
          (std::vector< std::pair<std::vector<double>, std::vector<double> > >
           (*)(std::vector<std::vector< std::pair<std::vector<double>, std::vector<double> > > >&,
               RXVec)) &combine_landscapes);
    m.def("calc_average_landscape", &calc_average_landscape);
    m.def("calc_landscape_norm", &calc_landscape_norm);
    m.def("calc_landscape_dist", &calc_landscape_dist);
    m.def("calc_dist_mat", &calc_dist_mat);
    m.def("calc_dist_mat_norms", &calc_dist_mat_norms);

    
    // Protein algorithms

    init_protein_templates<2>(m);
    init_protein_templates<3>(m);
    
    m.def("merge_basins", &merge_basins);
    

//     m.def("find_optimal_hinge", &find_optimal_hinge,
//           py::arg("V"), py::arg("coV"), py::arg("filt"), py::arg("comp"),
//           py::arg("n_basins")=2, py::arg("verbose")=false);
//     m.def("find_hinge_persistence_pairs", &find_hinge_persistence_pairs,
//           py::arg("pairs"), py::arg("V"), py::arg("coV"), py::arg("filt"), py::arg("comp"),
//           py::arg("n_basins")=2, py::arg("min_size")=1, py::arg("reset")=false, py::arg("verbose")=false);
//     m.def("simplify_morse_complex",
//           (void (*)(std::vector<std::pair<int, int> >&, RXiVec, RXiVec, Filtration&, CellComplex&, bool, bool))
//           &simplify_morse_complex,
//           py::arg("pairs"), py::arg("V"), py::arg("coV"), py::arg("filt"),
//           py::arg("comp"), py::arg("reset")=false, py::arg("verbose") = false);

    

};
