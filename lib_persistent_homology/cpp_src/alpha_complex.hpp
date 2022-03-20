#ifndef ALPHACOMPLEX_HPP
#define ALPHACOMPLEX_HPP


// d-dimensional triangulations
#include <CGAL/Epick_d.h>
// #include <CGAL/Delaunay_triangulation.h>
#include <CGAL/Regular_triangulation.h>

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <bitset>

#include "eigen_macros.hpp"
#include "embedding.hpp"
#include "cell_complex.hpp"
#include "filtration.hpp"
#include "search.hpp"
#include "space_partition.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

template <int DIM> CellComplex construct_alpha_complex(Embedding<DIM> &embed,
                                                       std::vector<double> &weights, bool oriented=false, int dim_cap=DIM) {


    // d-dimensional Kernel used to define Euclidean space (R^d)
    typedef CGAL::Epick_d< CGAL::Dimension_tag<DIM> > Kernel;

    // Triangulation data structure
    // Template info is explicitly defined in order to stick an integer label into each Triangulation_vertex
    typedef CGAL::Triangulation_data_structure<
                typename Kernel::Dimension,
                CGAL::Triangulation_vertex<CGAL::Regular_triangulation_traits_adapter<Kernel>, int>,
                CGAL::Triangulation_full_cell<CGAL::Regular_triangulation_traits_adapter<Kernel> > >
                    Triangulation_data_structure;

    // Regular Delaunay triangulation
    typedef CGAL::Regular_triangulation<Kernel, Triangulation_data_structure> Regular_triangulation;
    // d-dimensional point
    typedef typename Kernel::Point_d Point;
    // d-dimensional weighted point
    typedef typename Kernel::Weighted_point_d WPoint;


    // Triangulation object
    Regular_triangulation tri(DIM);

    // Map of lists of vertices of all simplices to index of simplex in alpha complex
    std::map<std::vector<int> , int> simplex_to_index;

    CellComplex alpha_comp(DIM, true, oriented);

    int NV = embed.NV;

    // Add all vertices to cell complex
    for(int i = 0; i < NV; i++) {

        DVec pos = embed.get_pos(i);
        std::vector<double> coords(pos.data(), pos.data()+DIM);
        WPoint w(Point(coords.begin(), coords.end()), weights[i]);

        auto vertex = tri.insert(w);
        
        if ((int)tri.number_of_vertices() < i+1) {
            py::print("Error: Incompatible vertices.");
            return CellComplex(DIM, true, oriented);
        }
        
        vertex->data() = i;

        std::vector<int> facets;
        std::vector<int> coeffs;
        alpha_comp.add_cell(i, 0, facets, coeffs);

        simplex_to_index.emplace(std::piecewise_construct, std::forward_as_tuple(1, i), std::forward_as_tuple(i));
    }

    if(embed.periodic) {


        // First create list of images represented by the offset from the main image
        // There will be 2^d images
        std::vector<double> offset{0, 1};

        std::vector<std::vector<double> > image_list(1, std::vector<double>());

        // Calculate cartesion product of offset
        for(int d = 0; d < DIM; d++) {
            std::vector<std::vector<double> > new_image_list;

            for(auto image: image_list) {
                for(auto off: offset) {

                    auto copy = image;
                    copy.push_back(off);

                    new_image_list.push_back(copy);

                }
            }

            image_list = new_image_list;

        }

        // Delete the main image as it already exists
        std::vector<double> zero_image(DIM, 0);
        image_list.erase(std::remove(image_list.begin(), image_list.end(), zero_image), image_list.end());

        // Add each of the additionalimages to the triangulation
        int vi = NV;
        for(auto image: image_list) {

            DVec offset = XVecMap(image.data(), DIM);
            
            for(int i = 0; i < NV; i++) {

                DVec pos = embed.box_mat * (embed.get_vpos(i) + offset);

                std::vector<double> coords(pos.data(), pos.data()+DIM);
                WPoint w(Point(coords.begin(), coords.end()), weights[i]);

                auto vertex = tri.insert(w);
                vertex->data() = vi;

                vi++;
            }
            
        }


    }



    // py::print("Regular triangulation successfully computed: " , tri.number_of_vertices(), " vertices, ",
    // tri.number_of_finite_full_cells()," finite cells.");

//     for(auto it = tri.finite_full_cells_begin(); it != tri.finite_full_cells_end(); it++) {
//         py::print("Cell:");

//         for(auto vit = it->vertices_begin(); vit != it->vertices_end(); vit++) {
//             // Need to dereference first, since vit is a pointer to a vertex handle
//             py::print((*vit)->data());

//         }

//     }
    
    std::vector<std::vector<int> >  valid_tris;
    
    if(embed.periodic) {
        
        
        // std::vector<int> test = {1263, 5621, 6341, 7209};
        
        // Map of triangles (defined by vertices) -> combination of images (up to periodic BCs) -> (# spanned lattice dimensions, orientations, count)
        std::map<std::vector<int> , std::map<std::vector<int>, std::tuple<int, int, int> > > tri_info;
        
        // Iterate through all max dim simplices (triangles, tetrahedra, etc.)
        for(auto it = tri.finite_full_cells_begin(); it != tri.finite_full_cells_end(); it++) {

            std::vector<int> verts;
            std::vector<int> images;
            for(auto vit = it->vertices_begin(); vit != it->vertices_end(); vit++) {
                // Need to dereference first, since vit is a pointer to a vertex handle
                int vi = (*vit)->data(); 
                verts.push_back(vi % NV);
                
                images.push_back(vi / NV);
                
            }

            // Sort verts to ensure consistency
            std::sort(verts.begin(), verts.end());
            
            // Sort images for consistency
            std::sort(images.begin(), images.end());
            
            // if(verts == test) {
            //     py::print(verts, images);
            // }
            
            // Binary representation of images
            std::vector<std::bitset<DIM> > bin_images;
            for(int im: images) {
                bin_images.emplace_back(im);
            }
            
            // Calcuate number of spanned lattice dimensions
            int spanned_dims = 0;
            for(int d = 0; d < DIM; d++) {
                
                int coord = bin_images[0][d];
                for(int vi = 1; vi < DIM+1; vi++) {
                    if(bin_images[vi][d] != coord) {
                        spanned_dims++;
                        break;
                    }
                }
            }
            
            // Check if triangle with these vertices exists
            if(tri_info.count(verts)) {
                
                
                // Check if periodic image of triangle exists
                // Iterate through number of different directions to translate, depending on number spanned dimensions
                
                bool exists = false;
                for(int d = 0; d <= DIM-spanned_dims; d++) {
                    
                    // Mask to pick out d direction to translate
                    std::vector<bool> mask(d, true);
                    mask.resize(DIM, false);
                    // Iterate through every permutation of translations in d directions
                    do {
                        
                        std::vector<int> translated_images(DIM+1);
                        
                        // Iterate through each vertex in triangle
                        for(int vi = 0; vi < DIM+1; vi++) {
                            std::bitset<DIM> bin_copy = bin_images[vi];
                            // Translate using binary representation
                            for(int m = 0; m < DIM; m++) {
                                if(mask[m]) {
                                    bin_copy.flip(m);
                                }
                            }
                            
                            translated_images[vi] = int(bin_copy.to_ulong());
                            
                        }
                        
                        
                        // Sort for consistency
                        std::sort(translated_images.begin(), translated_images.end());
                        
                        
                        // Check if this combination of images exists for this triangle
                        if(tri_info[verts].count(translated_images)) {
                            // If so, then increment count and break
                            std::get<2>(tri_info[verts][translated_images])++;
                            
                            // Check if untranslated image overlaps with previous observed image (meaning this image is actually a separate orientation)
                            for(auto vi: translated_images) {
                                
                                bool complete = false;
                                for(auto vj: images) {
                                    if(vi == vj) {
                                        std::get<1>(tri_info[verts][translated_images])++;
                                        complete = true;
                                        break;
                                    }
                                }
                                if(complete) {
                                    break;
                                }
                            }
                            
                            exists = true;
                            break;
                        }
                        
                        
                    } while(std::prev_permutation(mask.begin(), mask.end()));
                    
                    
                    if(exists) {
                        break;
                    }
                    
                }
                
                // If translated version was not found, then add
                if(!exists) {
                    tri_info[verts].emplace(images, std::forward_as_tuple(spanned_dims, 1, 1));
                }
                
                
            } else {
                // If triangle doesn't exist then add
                tri_info[verts].emplace(images, std::forward_as_tuple(spanned_dims, 1, 1));
                
            }
            
        }
            

        

        // Iterate through each found triangle
        // This is slightly less than optimal due to checking for validity.
        for(auto pair: tri_info) {

            auto verts = pair.first;
            
            bool is_valid = false;
            
            // Iterate through each possible version of the triangle
            for(auto tri_pair: pair.second) {
                
                auto images = tri_pair.first;
                int dist;
                int orientations;
                int count;
                std::tie(dist, orientations, count) = tri_pair.second;
                
                // if(test == verts) {
                //     py::print(verts, images, dist, orientations, count, count/orientations, "?=", int(pow(2, DIM-dist)));
                // }
                
                // If number of replicates/orientations = 2^(crossing dist - DIM), then is valid triangle
                if(count/orientations == int(pow(2, DIM-dist))) {
                    
                    if(orientations > 1) {
                        py::print("Warning: Valid triangle with more than one observed orientation. This is not a problem, but is worth reporting to Jason.");
                    }
                    
                    
                    if(!is_valid) {
                        is_valid = true;
                        valid_tris.push_back(verts);
                    } else{
                        py::print("Warning: More than one valid version of triangle present! Report to Jason.");
                        py::print("tri info:", verts, images, count, dist);
                    }
                    
                }
                
            }
            
            

        }


    } else {
        
        // Iterate through all max dim simplices (triangles, tetrahedra, etc.)
        for(auto it = tri.finite_full_cells_begin(); it != tri.finite_full_cells_end(); it++) {

            std::vector<int> verts;
            std::vector<DiVec> images;
            for(auto vit = it->vertices_begin(); vit != it->vertices_end(); vit++) {
                // Need to dereference first, since vit is a pointer to a vertex handle
                int vi = (*vit)->data(); 
                verts.push_back(vi % NV);
            }
            
            valid_tris.push_back(verts);

        } 
        
    }
    

    // Iterate through cells by dimension
    for(int d = 1; d <= dim_cap; d++) {

        int label = 0;

        // Iterate through all max dim simplices (triangles, tetrahedra, etc.)
        for(auto verts: valid_tris) {


            // Iterate through every size d+1 subset

            // A mask to pick out exactly d+1 verts
            std::vector<bool> mask(d+1, true);
            mask.resize(verts.size(), false);
            do {

                // Find verts in simplex
                std::vector<int> simplex;
                for(std::size_t j = 0; j < verts.size(); j++) {
                    if(mask[j]) {
                        simplex.push_back(verts[j]);
                    }
                }

                // Sorted list of vertices of cell
                std::sort(simplex.begin(), simplex.end());

                // If simplex already exists in graph complex, then skip
                if(simplex_to_index.count(simplex)) {
                    continue;
                }

                simplex_to_index[simplex] = alpha_comp.ncells;

                // Find facets
                std::vector<int> facets;
                std::vector<int> coeffs;
                for(std::size_t j = 0; j < simplex.size(); j++) {
                    std::vector<int> facet(simplex);
                    facet.erase(facet.begin()+j);
                    facets.push_back(simplex_to_index[facet]);
                    coeffs.push_back(-2*(j%2)+1);
                }
                alpha_comp.add_cell(label, d, facets, coeffs);
                label++;

            } while(std::prev_permutation(mask.begin(), mask.end()));

        }

    }

    alpha_comp.construct_cofacets();
    alpha_comp.make_compressed();

    return alpha_comp;

}

// template <int DIM> CellComplex construct_alpha_complex(Embedding<DIM> &embed,
//                                                        std::vector<double> &weights, bool oriented=false, int dim_cap=DIM) {


//     // d-dimensional Kernel used to define Euclidean space (R^d)
//     typedef CGAL::Epick_d< CGAL::Dimension_tag<DIM> > Kernel;

//     // Triangulation data structure
//     // Template info is explicitly defined in order to stick an integer label into each Triangulation_vertex
//     typedef CGAL::Triangulation_data_structure<
//                 typename Kernel::Dimension,
//                 CGAL::Triangulation_vertex<CGAL::Regular_triangulation_traits_adapter<Kernel>, int>,
//                 CGAL::Triangulation_full_cell<CGAL::Regular_triangulation_traits_adapter<Kernel> > >
//                     Triangulation_data_structure;

//     // Regular Delaunay triangulation
//     typedef CGAL::Regular_triangulation<Kernel, Triangulation_data_structure> Regular_triangulation;
//     // d-dimensional point
//     typedef typename Kernel::Point_d Point;
//     // d-dimensional weighted point
//     typedef typename Kernel::Weighted_point_d WPoint;


//     // Triangulation object
//     Regular_triangulation tri(DIM);

//     // Map of lists of vertices of all simplices to index of simplex in alpha complex
//     std::map<std::vector<int> , int> simplex_to_index;

//     CellComplex alpha_comp(DIM, true, oriented);

//     int NV = embed.NV;

//     // Add all vertices to cell complex
//     for(int i = 0; i < NV; i++) {

//         DVec pos = embed.get_pos(i);
//         std::vector<double> coords(pos.data(), pos.data()+DIM);
//         WPoint w(Point(coords.begin(), coords.end()), weights[i]);

//         auto vertex = tri.insert(w);
        
//         if ((int)tri.number_of_vertices() < i+1) {
//             py::print("Error: Incompatible vertices.");
//             return CellComplex(DIM, true, oriented);
//         }
        
//         vertex->data() = i;

//         std::vector<int> facets;
//         std::vector<int> coeffs;
//         alpha_comp.add_cell(i, 0, facets, coeffs);

//         simplex_to_index.emplace(std::piecewise_construct, std::forward_as_tuple(1, i), std::forward_as_tuple(i));
//     }

//     if(embed.periodic) {


//         // First create list of images represented by the offset from the main image
//         std::vector<double> offset{-1, 0, 1};

//         std::vector<std::vector<double> > image_list(1, std::vector<double>());

//         // Calculate cartesion product of offset
//         for(int d = 0; d < DIM; d++) {
//             std::vector<std::vector<double> > new_image_list;

//             for(auto image: image_list) {
//                 for(auto off: offset) {

//                     auto copy = image;
//                     copy.push_back(off);

//                     new_image_list.push_back(copy);

//                 }
//             }

//             image_list = new_image_list;

//         }

//         // Delete the main image as it already exists
//         std::vector<double> zero_image(DIM, 0);
//         image_list.erase(std::remove(image_list.begin(), image_list.end(), zero_image), image_list.end());

//         // Add each image to the triangulation
//         int vi = NV;
//         for(auto image: image_list) {

//             DVec offset = XVecMap(image.data(), DIM);

//             for(int i = 0; i < NV; i++) {

//                 DVec pos = embed.box_mat * (embed.get_vpos(i) + offset);

//                 std::vector<double> coords(pos.data(), pos.data()+DIM);
//                 WPoint w(Point(coords.begin(), coords.end()), weights[i]);

//                 auto vertex = tri.insert(w);
//                 vertex->data() = vi;

//                 vi++;
//             }

//         }


//     }



//     // py::print("Regular triangulation successfully computed: " , tri.number_of_vertices(), " vertices, ",
//     // tri.number_of_finite_full_cells()," finite cells.");

// //     for(auto it = tri.finite_full_cells_begin(); it != tri.finite_full_cells_end(); it++) {
// //         py::print("Cell:");

// //         for(auto vit = it->vertices_begin(); vit != it->vertices_end(); vit++) {
// //             // Need to dereference first, since vit is a pointer to a vertex handle
// //             py::print((*vit)->data());

// //         }

// //     }

//     // Iterate through each corner and add all higher-dimensional faces of corner simplices
//     for(int d = 1; d <= dim_cap; d++) {

//         int label = 0;

//         for(auto it = tri.finite_full_cells_begin(); it != tri.finite_full_cells_end(); it++) {

//             bool has_central_image = false;
//             std::vector<int> vertices;
//             for(auto vit = it->vertices_begin(); vit != it->vertices_end(); vit++) {
//                 // Need to dereference first, since vit is a pointer to a vertex handle
//                 // Mod by NV to take care of periodic BCs if turned on
//                 int vi = (*vit)->data();
//                 vertices.push_back(vi % NV);

//                 // Is at least one vertex contained in the central image?
//                 if(vi < NV) {
//                     has_central_image = true;
//                 }

//             }

//             // If none of the vertices are in the central image, then skip this cell
//             if(!has_central_image) {
//                 continue;
//             }

//             // Iterate through every size d+1 subset

//             // A mask to pick out exactly d+1 verts
//             std::vector<bool> mask(d+1, true);
//             mask.resize(vertices.size(), false);
//             do {

//                 // Find vertices in simplex
//                 std::vector<int> simplex;
//                 for(std::size_t j = 0; j < vertices.size(); j++) {
//                     if(mask[j]) {
//                         simplex.push_back(vertices[j]);
//                     }
//                 }

//                 // Sorted list of vertices of cell
//                 std::sort(simplex.begin(), simplex.end());

//                 // If simplex already exists in graph complex, then skip
//                 if(simplex_to_index.count(simplex)) {
//                     continue;
//                 }

//                 simplex_to_index[simplex] = alpha_comp.ncells;

//                 // Find facets
//                 std::vector<int> facets;
//                 std::vector<int> coeffs;
//                 for(std::size_t j = 0; j < simplex.size(); j++) {
//                     std::vector<int> facet(simplex);
//                     facet.erase(facet.begin()+j);
//                     facets.push_back(simplex_to_index[facet]);
//                     coeffs.push_back(-2*(j%2)+1);
//                 }
//                 alpha_comp.add_cell(label, d, facets, coeffs);
//                 label++;

//             } while(std::prev_permutation(mask.begin(), mask.end()));

//         }

//     }

//     // }

//     alpha_comp.construct_cofacets();
//     alpha_comp.make_compressed();

//     return alpha_comp;

// }


template <int DIM> double calc_power_distance(double alpha, DVec a, DVec pos, double weight=0.0) {
    return (a - pos).squaredNorm() - weight - alpha;
}


template <int DIM> std::tuple<double, DVec > calc_circumsphere(std::vector<int> &vertices,
                                                               Embedding<DIM> &embed, std::vector<double> &weights) {



    // Find vertex positions relative to first vertex
    XMat X = XMat::Zero(DIM, vertices.size());
    for(std::size_t i = 0; i < vertices.size(); i++) {
        X.block<DIM, 1>(0, i) = embed.get_vpos(vertices[i]);

        if(embed.periodic && i > 0) {
            DVec bvec = X.block<DIM, 1>(0, i) - X.block<DIM, 1>(0, 0);

            for(int d = 0; d < DIM; d++) {
                if(std::fabs(bvec(d)) > 0.5) {
                    bvec(d) -= ((bvec(d) > 0) - (bvec(d) < 0));
                }
            }

            X.block<DIM, 1>(0, i) = X.block<DIM, 1>(0, 0) + bvec;

        }

    }

    // Scale vertex positions to box size and shape
    for(std::size_t i = 0; i < vertices.size(); i++) {
        X.block<DIM, 1>(0, i) = embed.box_mat * X.block<DIM, 1>(0, i);
    }

    if(vertices.size() == 1) {
        return std::make_tuple(0.0, DVec::Zero());
    } else if(vertices.size() == DIM + 1) {

        XMat A = XMat::Zero(DIM+1, DIM+1);

        A.block<DIM+1, DIM>(0, 0) = 2.0 * X.block<DIM, DIM+1>(0, 0).transpose();

        A.block<DIM+1,1>(0, DIM) = XVec::Ones(DIM+1);

        XVec b = XVec::Zero(DIM+1);
        for(std::size_t i = 0; i < DIM+1; i++) {
            int vi = vertices[i];

            b(i) = X.block<DIM, 1>(0, i).squaredNorm() - weights[vi];
        }

        XVec x = A.partialPivLu().solve(b);

        return std::make_tuple(x(DIM) + x.segment<DIM>(0).squaredNorm(), x.segment<DIM>(0));

    } else {

        XMat A = XMat::Zero(DIM+vertices.size(), DIM+vertices.size());

        A.block(0, 0, vertices.size(), DIM) = 2.0 * X.block(0, 0, DIM, vertices.size()).transpose();

        A.block(0, DIM, vertices.size(), 1) = XVec::Ones(vertices.size());

        DVec v0 = X.block<DIM, 1>(0, 0);
        for(std::size_t i = 1; i < vertices.size(); i++) {
            A.block<DIM,1>(vertices.size(), DIM+i) = v0 - X.block<DIM, 1>(0, i);
        }

        A.block<DIM, DIM>(vertices.size(), 0) = DMat::Identity();

        XVec b = XVec::Zero(DIM+vertices.size());
        for(std::size_t i = 0; i < vertices.size(); i++) {
            int vi = vertices[i];
            b(i) = X.block<DIM, 1>(0, i).squaredNorm() - weights[vi];
        }

        b.segment<DIM>(vertices.size()) = v0;

        XVec x = A.partialPivLu().solve(b);

        return std::make_tuple(x(DIM) + x.segment<DIM>(0).squaredNorm(), x.segment<DIM>(0));


    }

}

template <int DIM> XVec calc_alpha_vals(CellComplex &comp, Embedding<DIM> &embed,
                                                       std::vector<double> &weights, double alpha0 = -1.0) {


    DMat box_mat_inv = embed.box_mat.inverse();

    XVec alpha_vals = XVec::Zero(comp.ncells);

    for(int c = comp.ncells-1; c >= 0; c--) {

        // Skip vertices
        if(comp.get_dim(c) == 0) {
            alpha_vals[c] = alpha0;
            continue;
        }

        std::unordered_set<int> verts = comp.get_faces(c, 0);

        double alpha;
        DVec a;
        std::vector<int> tmp(verts.begin(), verts.end());
        std::tie(alpha, a) = calc_circumsphere<DIM>(tmp, embed, weights);

        alpha_vals(c) = alpha;

        // Skip highest dimension triangles
        if(comp.get_dim(c) == DIM) {
            continue;
        }

        // For a Delaunay triangulation, only simplices of dimension DIM must have empty circumspheres
        // Some simplices between dimension 0 and DIM don't appear in the alpha shape filtration at the value of their particular alpha
        // This is because for a particular value of alpha, each simplex follows one of two rules:
        // 1. has an empty circumsphere or
        // 2. is the face of another simplex that is part of the alpha complex (see 1)
        // We must detect simplices that don't obey #1 and change their value of alpha to that of their youngest coface

        // Find cofaces of dimension DIM
        auto cofaces = comp.get_cofaces(c, DIM);

        bool conflict = false;
        // For each coface get the list of its vertices and check if any fall inside the circumsphere of c
        for(auto cf: cofaces) {

            std::unordered_set<int> coface_verts = comp.get_faces(cf, 0);

            for(auto vi: coface_verts) {

                // If inside the circumsphere, mark as conflict and continue
                DVec x = embed.get_vpos(vi);

                if(embed.periodic) {
                    DVec bvec = x - box_mat_inv * a;

                    for(int d = 0; d < DIM; d++) {
                        if(std::fabs(bvec(d)) > 0.5) {
                            bvec(d) -= ((bvec(d) > 0) - (bvec(d) < 0));
                        }
                    }

                    x = a + embed.box_mat*bvec;
                } else {
                    
                    x = embed.box_mat * x;
                }

                if(!verts.count(vi) && calc_power_distance(alpha, a, x, weights[vi]) < 0.0) {
                    conflict = true;
                    break;
                }
            }

            if(conflict) {
                break;
            }

        }


        if(!conflict) {
            continue;
        }

        // If this simplex poses a conflict, then set it's value of alpha to it's youngest cofacet
        auto cofacets = comp.get_cofacets(c);
        double conflict_alpha = 1e10;
        for(auto cf: cofacets) {
            if(alpha_vals(cf) < conflict_alpha) {
                conflict_alpha = alpha_vals(cf);
            }
        }

        alpha_vals(c) = conflict_alpha;


    }

    return alpha_vals;

}


template <int DIM> Filtration construct_alpha_filtration(CellComplex &comp, Embedding<DIM> &embed,
                                                         std::vector<double> &weights, double alpha0 = -1.0) {

    auto alpha_vals = calc_alpha_vals<DIM>(comp, embed, weights, alpha0);

    return construct_filtration(comp, alpha_vals);

}

//////////////////////////////////////////////////////////////////////////
//Simplification/Pruning
//////////////////////////////////////////////////////////////////////////

template <int DIM> CellComplex join_dtriangles(CellComplex &comp, RXVec alpha_vals, double threshold=0.0) {

    std::vector<int> disjoint_set(comp.ncells);
    std::iota(disjoint_set.begin(), disjoint_set.end(), 0);


    CellComplex simp_comp(DIM, true, false);

    std::vector<int> full_to_reduced(comp.ncells, -1);
    for(int c = 0; c < comp.dcell_range[DIM].first; c++) {

        // Find all edges
        auto edges = comp.get_faces(c, 1);

        // Check if any edge has alpha value larger than threshold
        bool prune = false;
        for(auto ei: edges) {
            if(alpha_vals[ei] > threshold) {
                prune = true;
                disjoint_set[c] = -1;
                break;
            }
        }

        if(prune && comp.get_dim(c) == DIM - 1) {
            // Join cofacets
            std::vector<int> cofacets = comp.get_cofacets(c);

            std::sort(cofacets.begin(), cofacets.end());
            for(std::size_t i = 1; i < cofacets.size(); i++) {

                int rooti = cofacets[i];
                while(disjoint_set[rooti] != rooti) {
                    rooti = disjoint_set[rooti];
                }


                int root0 = cofacets[0];
                while(disjoint_set[root0] != root0) {
                    root0 = disjoint_set[root0];
                }

                if(root0 < rooti) {
                    disjoint_set[rooti] = root0;
                } else {
                    disjoint_set[root0] = rooti;
                }


            }

//             if(c == 16372) {
//                 py::print("heyo", cofacets);

//                 for(auto cf: cofacets) {
//                     py::print(cf, disjoint_set[cf]);
//                 }
//             }


        }

        if(prune) {
            continue;
        }

        // If not pruning, then add to cell complex
        std::vector<int> facets;
        auto frange = comp.get_facet_range(c);
        for(auto it=frange.first; it!= frange.second; it++) {
            facets.push_back(full_to_reduced[*it]);
        }

        std::vector<int> coeffs;
        auto crange = comp.get_coeff_range(c);
        for(auto it=crange.first; it!= crange.second; it++) {
            coeffs.push_back(*it);
        }

        full_to_reduced[c] = simp_comp.ncells;

        simp_comp.add_cell(simp_comp.ndcells[comp.get_dim(c)], comp.get_dim(c), facets, coeffs);

    }

    std::map<int, std::set<int> > tris;
    for(int c = comp.dcell_range[DIM].first; c < comp.dcell_range[DIM].second; c++) {

        int root = c;
        while(disjoint_set[root] != root) {
            root = disjoint_set[root];
        }

//         if(c == 24549 || c == 24554) {
//             py::print(c, "root", root, disjoint_set[c]);
//         }

        tris[root].insert(c);

    }

    for(auto pair: tris) {

        int c = pair.first;

//         std::unordered_set<int> facet_set;
        std::unordered_map<int, int> facet_set;
        for(auto alpha: pair.second) {

            // This might need to be implemented with symmetric difference addition
            // However, this seems to result in removed edges being incorrectly added as facets
            // Might be better to deal with coefficients
            for(auto beta: comp.get_facets(alpha)) {

                if(facet_set.count(beta)) {
                    facet_set[beta]++;
                } else {
                    facet_set[beta] = 1;
                }

//                 if(disjoint_set[beta] != -1) {
//                     facet_set.insert(beta);
//                 }
            }

        }

//         if(c == 24549 || c == 24554) {
//             py::print(c, facet_set);
//         }

        std::vector<int> facets;
        for(auto pair: facet_set) {


            if(pair.second % 2 == 0) {
                continue;
            }

            int alpha = pair.first;


            if(full_to_reduced[alpha] == -1) {
//             if(alpha == 16372) {
                py::print("help!", alpha, pair.second, alpha_vals[alpha], py::arg("flush")=true);

                for(auto beta: comp.get_cofacets(alpha)) {
                    py::print("cofacet", beta, comp.get_facets(beta));
                }
            }

            facets.push_back(full_to_reduced[alpha]);
        }

        // Coefficients are not treated here
        std::vector<int> coeffs;



        full_to_reduced[c] = simp_comp.ncells;

        simp_comp.add_cell(simp_comp.ndcells[comp.get_dim(c)], comp.get_dim(c), facets, coeffs);


    }

    simp_comp.construct_cofacets();
    simp_comp.make_compressed();

    return simp_comp;


}



// // Calculate condition numbers of triangulation Jacobians
// template <int DIM> XVec calc_flatness(CellComplex &comp, Embedding<DIM> &embed) {


//     XVec flatness = XVec::Zero(comp.ndcells[DIM]);

//     for(int c = comp.dcell_range[DIM].first; c < comp.ncells; c++) {

//         auto vset = comp.get_faces(c, 0);

//         std::vector<int> verts(vset.begin(), vset.end());

//         int vi = verts[0];

//         DVec O = embed.get_vpos(vi);
        
//         DMat X = DMat::Zero();

//         for(int m = 0; m < DIM; m++) {

//             int vj = verts[1+m];

//             DVec bvec = embed.get_vpos(vj) - O;
            
//             for(int d = 0; d < DIM; d++) {
//                 if(std::fabs(bvec(d)) > 0.5) {
//                     bvec(d) -= ((bvec(d) > 0) - (bvec(d) < 0));
//                 }
//             }

//             bvec = embed.box_mat * bvec;

//             X += bvec * bvec.transpose();

//         }
        
        
//         Eigen::SelfAdjointEigenSolver<DMat> esolver(X);
        
//         DVec evals = esolver.eigenvalues();

//         flatness(comp.get_label(c)) = evals[DIM-1] / evals[0];
                
        
//     }
        

//     return flatness;




// }


//////////////////////////////////////////////////////////////////////////
//Cell type counting
//////////////////////////////////////////////////////////////////////////

std::unordered_map<int, std::unordered_map<int, std::unordered_map<std::string, int> > >
    calc_radial_edge_counts(std::vector<int> &cell_list, std::vector<std::string> &edge_types, 
                            CellComplex &comp, int max_dist=-1) {
    
    // particle->dist->edge_type->count
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<std::string, int> > > edge_count;

    for(int c: cell_list) {

        // Always search one extra so that reach both vertices of each edge
        auto dists = calc_comp_point_dists(c, comp, max_dist + 1);

        for(int i = comp.dcell_range[1].first; i < comp.dcell_range[1].second; i++) {
            if(dists[i] <= 0) {
                continue;
            }
                
            auto verts = comp.get_facets(i);
            
            int vi = verts[0];
            int vj = verts[1];
            
            if(dists[vi]==-1 || dists[vj]==-1) {
                continue;
            }
            
//             edge_count[c][dists[i]][edge_types[comp.get_label(i)]]++;
            edge_count[c][dists[vi]/2 + dists[vj]/2][edge_types[comp.get_label(i)]]++;
        }

    }

    return edge_count;


}

template <int DIM> std::unordered_map<int, std::tuple<std::map<int, int>, std::map<int, int> > >
    calc_radial_gap_distribution(std::vector<int> &cell_list, CellComplex &comp, Embedding<DIM> &embed, std::vector<double> &rad, int max_dist=-1, bool verbose=false) {

    // particle->(gaps[dist]->count, overlaps[dist]->count)
    std::unordered_map<int, std::tuple<std::map<int, int>, std::map<int, int> > > gap_distribution;

    int index = 0;
    for(auto c: cell_list) {

        if(verbose && index % 500 == 0) {
            py::print(index, "/", cell_list.size(), py::arg("flush")=true);
        }

        index++;

        std::map<int, int> gaps;
        std::map<int, int> overlaps;

        auto dists = calc_comp_point_dists(c, comp, max_dist);

        for(int i = comp.dcell_range[1].first; i < comp.dcell_range[1].second; i++) {
            if(dists[i] <= 0) {
                continue;
            }

            auto verts = comp.get_facets(i);
            
            int vi = verts[0];
            int vj = verts[1];
            
            double ri = rad[vi];
            double rj = rad[vj];
            
            double l0 = embed.get_diff(embed.get_vpos(vi), embed.get_vpos(vj)).norm();
            
            if( l0 > ri + rj ) {
                gaps[dists[i]]++;
            } else {
                overlaps[dists[i]]++;
            }
            
        }

        gap_distribution[c] = std::make_tuple(gaps, overlaps);

    }

    return gap_distribution;


}


template <int DIM> std::map<int,std::string> calc_gap_angle_class(std::vector<int> &cell_list, std::vector<double> &alpha_vals, CellComplex &comp, Embedding<DIM> &embed, int num_angles=128 , bool verbose=false) {

    // particle->(gaps[dist]->count, overlaps[dist]->count)
    std::map<int,std::string> class_map;
    for(int i = 0; i <comp.ndcells[0]; i++) {
        auto edges = comp.get_cofaces(i, 1);

        std::vector<int> gap_angles;

        for(auto e: edges) {
            if(alpha_vals[e] > 0.0) { //this is a gap
              auto edge_vertices = comp.get_faces(e,0);
              int j = *std::find_if(edge_vertices.begin(),edge_vertices.end(),[i](int k) {return k != i;});
              //now determine vector and thereby angle
              DVec v = embed.get_diff(embed.get_vpos(i), embed.get_vpos(j));
              //double s = 1.0/std::sqrt(2);
              double theta = std::atan2(v[1]-v[0],v[0] + v[1]);
              if (theta < 0.0)
                  theta +=2.0*M_PI;

              int itheta = theta*num_angles / (2.0*M_PI);
              gap_angles.push_back(itheta);
             }
        }
        //now we need to map to the single representative of our equivalence class
        //currently only works in 2d
        //representative = member of equivalence class with smallest small angle, smallest second small angle, etc...
        auto x_reflect = [num_angles](int theta) {return num_angles - 1 - theta;};
        auto y_reflect = [num_angles](int theta) {return (theta < num_angles/2 )? num_angles/2 - 1 -theta :3*num_angles/2 - 1 - theta ; };
        std::vector< std::vector<int> > equiv_class(4, gap_angles);
        std::transform(gap_angles.begin(),gap_angles.end(),equiv_class[1].begin(),x_reflect);
        std::transform(gap_angles.begin(),gap_angles.end(),equiv_class[2].begin(),y_reflect);
        std::transform(equiv_class[1].begin(),equiv_class[1].end(),equiv_class[3].begin(),y_reflect);
        for (int k = 0; k < 4; k++) {  std::sort(equiv_class[k].begin(),equiv_class[k].end());}

        auto lex_min = [](std::vector<int> v, std::vector<int> w) {
          return (std::lexicographical_compare(v.begin(),v.end(),w.begin(),w.end()) ? v : w );
        };
	//accumulate = reduce but exists before c++17
        std::vector<int> class_rep = std::accumulate(equiv_class.begin(),equiv_class.end(),equiv_class[0],lex_min);
        //now make a string for our representative
        std::string rep_string = "";
        for (auto it = class_rep.begin(); it!= class_rep.end(); it++) {
          std::string temp; 
          std::stringstream ss;
          ss << *it;
          ss >> temp;
          while(temp.length() < 3) {temp = "0" + temp;}
          rep_string = rep_string + temp;
        }
        if (rep_string == "064") {
	  py::print("064");
          py::print(equiv_class);
	}
        class_map[i] = rep_string;
    }

    return class_map;


}


std::unordered_map<int, std::map<int, std::map<int, int> >  >
    calc_radial_tri_distribution(std::vector<int> &cell_list, std::vector<double> &alpha_vals, CellComplex &comp, int max_dist=-1, bool verbose=false) {

    // particle->dist->triangle_type->count
    std::unordered_map<int, std::map<int, std::map<int, int> > > tri_distribution;

    int index = 0;
    for(auto c: cell_list) {

        if(verbose && index % 500 == 0) {
            py::print(index, "/", cell_list.size(), py::arg("flush")=true);
        }

        index++;

        std::map<int, std::map<int, int> > tris;

        auto dists = calc_comp_point_dists(c, comp, max_dist);

        for(int i = comp.dcell_range[comp.dim].first; i < comp.dcell_range[comp.dim].second; i++) {
            if(dists[i] <= 0) {
                continue;
            }

            auto edges = comp.get_faces(i, 1);

            int gap_count = 0;

            for(auto e: edges) {
                if(alpha_vals[e] > 0.0) {
                    gap_count++;
                }
            }

            tris[dists[i]][gap_count]++;
        }

        tri_distribution[c] = tris;

    }

    return tri_distribution;


}

std::unordered_map<int, std::tuple<std::map<int, std::map<int, int> >,
                                    std::map<int, std::map<int, int> >,
                                    std::map<int, std::map<int, int> > > >
    calc_angular_gap_distribution(std::vector<int> &cell_list, std::vector<double> &alpha_vals, CellComplex &comp, int max_dist=-1, bool verbose=false) {

    std::unordered_map<int, std::tuple<std::map<int, std::map<int, int> >,
                                    std::map<int, std::map<int, int> >,
                                    std::map<int, std::map<int, int> > > > gap_distribution;

    int index = 0;
    for(auto c: cell_list) {

        if(verbose && index % 500 == 0) {
            py::print(index, "/", cell_list.size(), py::arg("flush")=true);
        }

        index++;

        std::map<int, std::map<int, int> > gap_gap;
        std::map<int, std::map<int, int> > overlap_overlap;
        std::map<int, std::map<int, int> > gap_overlap;

        auto dists = calc_comp_point_dists(c, comp, max_dist);

        int max_rad_dist;
        if(max_dist == -1) {
            max_rad_dist = *std::max_element(dists.begin(), dists.end());
        } else {
            max_rad_dist = max_dist;
        }

        std::unordered_map<int, std::unordered_set<int> > dist_sets;
        for(int i = 0; i < comp.ncells; i++) {
            if(dists[i] <= max_rad_dist) {
                dist_sets[dists[i]].insert(i);
            }
        }

        for(int i = 1; i <= max_rad_dist; i++) {

            for(auto a: dist_sets[i]) {

                if(comp.get_dim(a) != 1) {
                    continue;
                }

                double alphaa = alpha_vals[a];

                auto layer_dists = calc_comp_point_dists_search_zone(a, dist_sets[i], comp);

                for(auto pair: layer_dists) {

                    int b = pair.first;

                    if(comp.get_dim(b) != 1) {
                        continue;
                    }

                    double alphab = alpha_vals[b];

                    if(alphaa > 0.0 && alphab > 0.0) {
                        gap_gap[i][pair.second/2]++;
                    } else if(alphaa < 0.0 && alphab < 0.0) {
                        overlap_overlap[i][pair.second/2]++;
                    } else {
                        gap_overlap[i][pair.second/2]++;
                    }

                }

            }

        }

        for(auto rad_pair: gap_gap) {
            for(auto ang_pair: rad_pair.second) {
                if(ang_pair.first != 0) {
                    gap_gap[rad_pair.first][ang_pair.first] /= 2;
                }
            }
        }

        for(auto rad_pair: overlap_overlap) {
            for(auto ang_pair: rad_pair.second) {
                if(ang_pair.first != 0) {
                    overlap_overlap[rad_pair.first][ang_pair.first] /= 2;
                }
            }
        }

        for(auto rad_pair: gap_overlap) {
            for(auto ang_pair: rad_pair.second) {
                gap_overlap[rad_pair.first][ang_pair.first] /= 2;
            }
        }

        gap_distribution[c] = std::make_tuple(gap_gap, overlap_overlap, gap_overlap);


    }

    return gap_distribution;


}




std::unordered_map<int,  std::unordered_map<int, std::map<std::tuple<std::string, std::string >, int> > >
    calc_radial_cycle_distribution(std::vector<int> &cell_list, std::vector<double> &alpha_vals, std::vector<std::string> &vertex_types, CellComplex &comp, int max_dist=-1, bool verbose=false) {


    // particle->dist->(vertex_types, edge_types)->count
    std::unordered_map<int,  std::unordered_map<int, std::map<std::tuple<std::string, std::string >, int> > > cycle_distribution;

    int index = 0;
    for(auto c: cell_list) {

        if(verbose && index % 500 == 0) {
            py::print(index, "/", cell_list.size(), py::arg("flush")=true);
        }

        index++;


        auto dists = calc_comp_point_dists(c, comp, max_dist);

        for(int i = 0; i < comp.ncells; i++) {
            if(dists[i] <= 0) {
                continue;
            }


            auto verts = comp.get_faces(i, 0);
            auto edges = comp.get_faces(i, 1);

            std::vector<std::string> vtypes_list;
            for(auto v: verts) {
                if(v == c) {
                    vtypes_list.push_back("t");
                } else {
                    vtypes_list.push_back(vertex_types[v]);
                }
            }

            std::sort(vtypes_list.begin(), vtypes_list.end(), std::greater<std::string>());
            std::string vtypes = std::accumulate(vtypes_list.begin(), vtypes_list.end(), std::string(""));

            std::vector<std::string> etypes_list;
            for(auto e: edges) {

                double alpha = alpha_vals[e];
                if(alpha > 0.0) {
                    continue;
                }

                std::vector<std::string> elabel_list;
                auto everts = comp.get_facets(e);
                for(auto ev: everts) {

                    if(ev == c) {
                        elabel_list.push_back("t");
                    } else {
                        elabel_list.push_back(vertex_types[ev]);
                    }
                }

                std::sort(elabel_list.begin(), elabel_list.end(), std::greater<std::string>());
                std::string elabel = std::accumulate(elabel_list.begin(), elabel_list.end(), std::string(""));

                etypes_list.push_back(elabel);

            }

            std::sort(etypes_list.begin(), etypes_list.end(), std::greater<std::string>());
            std::string etypes = std::accumulate(etypes_list.begin(), etypes_list.end(), std::string(""));


            cycle_distribution[c][dists[i]][std::forward_as_tuple(vtypes, etypes)]++;

        }

    }

    return cycle_distribution;


}





// This is here because it relies on betti numbers

std::tuple<CellComplex, std::unordered_set<int> >  
    prune_cell_complex_sequential_surface(CellComplex &comp, RXVec priority, std::unordered_set<int> &preserve,
                                          std::unordered_set<int> &surface, bool preserve_stop=true,
                                          bool allow_holes=false, double threshold=0.0, int target_dim=-1, bool verbose=false) {
    
    
    // Only works for cells of maximum dimension
    if(target_dim==-1) {
        target_dim = comp.dim;
    }
    
    
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int> >> PQ;
    
    for(int c = comp.dcell_range[target_dim].first; c < comp.dcell_range[target_dim].second; c++) {
        PQ.emplace(priority(comp.get_label(c)), c);
    }
    
    if(verbose) {
        py::print(PQ.size(), PQ.top().first, py::arg("flush")=true);
    }
    
    std::unordered_set<int> rem_cells;
    while(!PQ.empty()) {
        
        if(verbose && PQ.size() % 100 == 0) {
            py::print(PQ.size(), PQ.top().first, py::arg("flush")=true);
        }
       
        auto top = PQ.top();
        PQ.pop();
        
        double val = top.first;
        int c = top.second;
        
         
        
        
                        
        if(val <= threshold) {
            break;
        }
        

        std::unordered_set<int> test_cells;
        test_cells.insert(c);
        
        bool stop = false;
        for(int f: comp.get_faces(c)) {
        
            auto cofaces = comp.get_cofaces(f, target_dim);
            bool has_coface = false;
            for(int cf: cofaces) {
                if(!rem_cells.count(cf) && !test_cells.count(cf)) {
                    has_coface = true;
                    break;
                }
            }

            if(!has_coface) {
                
                if(preserve.count(f)) {
                    stop = true;
                    break;
                } else {
                    test_cells.insert(f);
                }
            }
                
        }
        
        if(stop && preserve_stop) {
            break;
        } else if(stop && !preserve_stop) {
            continue;
        }
        
        if(!allow_holes) {

            std::unordered_set<int> all_rem_cells(rem_cells);
            all_rem_cells.insert(test_cells.begin(), test_cells.end());
            std::vector<int> all_rem_cells_list(all_rem_cells.begin(), all_rem_cells.end());
            CellComplex comp_tmp = prune_cell_complex(all_rem_cells_list, comp);

            auto betti = calc_betti_numbers(comp_tmp);
            
            if(betti[0] > 1) {
                break;
            }
            for(std::size_t i = 1; i < betti.size(); i++) {
                if(betti[i] > 0) {
                    stop = true;
                }
            }

            if(stop) {
                break;
            }
        }
                
        rem_cells.insert(test_cells.begin(), test_cells.end());
        
        auto facets = comp.get_facets(c);
        bool is_surface = false;
        for(int f: facets) {
            if(surface.count(f)) {
                is_surface=true;
                break;
            }
        }

        if(is_surface) {
            for(int f: facets) {
                if(surface.count(f)) {
                    surface.erase(f);
                } else {
                    surface.insert(f);
                }
            }
        }
            
        
    }
    
    std::vector<int> rem_cells_list(rem_cells.begin(), rem_cells.end());
    
    auto result = prune_cell_complex_map(rem_cells_list, comp);
    
    CellComplex comp_tmp = std::get<0>(result);
    auto full_to_reduced = std::get<1>(result);
    
    std::unordered_set<int> red_surface;
    for(int i: surface) {
        red_surface.insert(full_to_reduced[i]);
    }
        
    return std::make_tuple(comp_tmp, red_surface);
    
    
}

CellComplex prune_cell_complex_sequential(CellComplex &comp, RXVec priority, std::unordered_set<int> &preserve, bool preserve_stop=true,
                                          bool allow_holes=false, double threshold=0.0, int target_dim=-1) {
    
    std::unordered_set<int> surface;
    auto result = prune_cell_complex_sequential_surface(comp, priority, preserve, surface, preserve_stop, allow_holes, threshold, target_dim);

    return  std::get<0>(result);
    
    
}



 
    
#endif //ALPHACOMPLEX_HPP

