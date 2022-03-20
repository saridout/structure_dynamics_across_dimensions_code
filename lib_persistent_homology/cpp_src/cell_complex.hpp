#ifndef CELLCOMPLEX_HPP
#define CELLCOMPLEX_HPP

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <vector>
#include <unordered_map>
#include <utility>
#include <queue>

class CellComplex {


public:

    int dim;
    int ncells;
    std::vector<int> ndcells;
    std::vector<std::pair<int,int> > dcell_range;
    bool regular;
    bool oriented;



protected:

    std::vector<int> labels;
    std::vector<int> dims;
    std::vector<int> facet_ind;
    std::vector<int> facets;
    std::vector<int> cofacet_ind;
    std::vector<int> cofacets;
    std::vector<int> coeffs;

public:

    CellComplex(int dim, bool regular = true, bool oriented = false) :
        dim(dim), ncells(0), ndcells(dim+1, 0), dcell_range(dim+1, std::pair<int,int>(0,0)), regular(regular), oriented(oriented) {
        facet_ind.push_back(0);
    }


    // Add a cell to the cell complex
    void add_cell(int label, int dim, std::vector<int> &cell_facets, std::vector<int> &cell_coeffs) {

        labels.push_back(label);
        dims.push_back(dim);
        facet_ind.push_back(facet_ind[ncells] + cell_facets.size());
        facets.insert(facets.end(), cell_facets.begin(), cell_facets.end());
        if(!regular || oriented) {
            coeffs.insert(coeffs.end(), cell_coeffs.begin(), cell_coeffs.end());
        }

        ncells++;

        ndcells[dim]++;
        dcell_range[dim].second++;
        for(int d = dim+1; d < this->dim+1; d++) {
            dcell_range[d].first++;
            dcell_range[d].second++;
        }

    }


    // Get cell dimension
    int get_dim(int alpha) {
        return dims[alpha];
    }

    // Get cell label
    int get_label(int alpha) {
        return labels[alpha];
    }

    // Get facets of cell
    std::vector<int> get_facets(int alpha) {
        return std::vector<int>(facets.begin()+facet_ind[alpha], facets.begin()+facet_ind[alpha+1]);
    }

    // Get cofaces of cell
    std::vector<int> get_cofacets(int alpha) {
        return std::vector<int>(cofacets.begin()+cofacet_ind[alpha], cofacets.begin()+cofacet_ind[alpha+1]);
    }

    // Get coefficients of facets of cell
    std::unordered_map<int, int> get_coeffs(int alpha) {

        std::unordered_map<int, int> coeff_map;

        for(int i = facet_ind[alpha]; i < facet_ind[alpha+1]; i++) {
            if(oriented || !regular) {
                coeff_map[facets[i]] = coeffs[i];
            } else {
                coeff_map[facets[i]] = 1;
            }
        }

        return coeff_map;

    }

    // Get faces of all dimensions of cell
    std::unordered_set<int> get_faces(int alpha, int target_dim=-1) {

        return get_star(alpha, true, target_dim);

    }

    // Get cofaces of all dimensions of cell
    std::unordered_set<int> get_cofaces(int alpha, int target_dim=-1) {

        return get_star(alpha, false, target_dim);

    }

    // Get star or costar of cell (useful for finding cofaces or faces)
    std::unordered_set<int> get_star(int alpha, bool dual, int target_dim=-1) {

        std::unordered_set<int> star;

        std::unordered_set<int> seen;
        seen.insert(alpha);

        std::queue<int> Q;
        Q.push(alpha);





        while(!Q.empty()) {
            int a = Q.front();
            Q.pop();

             if(target_dim == -1 || get_dim(a) == target_dim) {
                star.insert(a);
            }

            // star is cofaces and costar is faces
            auto range = dual ? get_facet_range(a) : get_cofacet_range(a);
            for(auto it = range.first; it != range.second; it++) {
                int b = *it;
                if(!seen.count(b)) {
                    Q.push(b);
                    seen.insert(b);
                }
            }

        }

        return star;

    }

    // Get all labels for list of cells
    std::unordered_set<int> get_labels(std::unordered_set<int> &cell_list) {

        std::unordered_set<int> cell_labels;

        for(auto alpha: cell_list) {
            cell_labels.insert(labels[alpha]);
        }

        return cell_labels;
    }




    // Utility function to add cell using ranges of cells (only useful in C++)
    void add_cell(int label, int dim,
                  std::pair<std::vector<int>::iterator, std::vector<int>::iterator> &facet_range,
                  std::pair<std::vector<int>::iterator, std::vector<int>::iterator> &coeff_range) {

        labels.push_back(label);
        dims.push_back(dim);

        int delta_size = facets.size();
        facets.insert(facets.end(), facet_range.first, facet_range.second);
        delta_size = facets.size() - delta_size;
        if(!regular || oriented) {
            coeffs.insert(coeffs.end(), coeff_range.first, coeff_range.second);
        }

        facet_ind.push_back(facet_ind[ncells] + delta_size);

        ncells++;

        ndcells[dim]++;
        dcell_range[dim].second++;
        for(int d = dim+1; d < this->dim+1; d++) {
            dcell_range[d].first++;
            dcell_range[d].second++;
        }

    }


    // Get facets of cell as range
    std::pair<std::vector<int>::iterator, std::vector<int>::iterator> get_facet_range(int alpha) {
        return std::make_pair(facets.begin()+facet_ind[alpha], facets.begin()+facet_ind[alpha+1]);
    }

    // Get cofacets of cell as range
    std::pair<std::vector<int>::iterator, std::vector<int>::iterator> get_cofacet_range(int alpha) {
        return std::make_pair(cofacets.begin()+cofacet_ind[alpha], cofacets.begin()+cofacet_ind[alpha+1]);
    }

    std::pair<std::vector<int>::iterator, std::vector<int>::iterator> get_coeff_range(int alpha) {
        if(oriented || !regular) {
            return std::make_pair(coeffs.begin()+facet_ind[alpha], coeffs.begin()+facet_ind[alpha+1]);
        } else {
            return std::make_pair(coeffs.begin(), coeffs.begin());
        }
    }

    // Compress sizes of vectors
    void make_compressed() {
        labels.shrink_to_fit();
        dims.shrink_to_fit();
        facet_ind.shrink_to_fit();
        facets.shrink_to_fit();
        coeffs.shrink_to_fit();
        cofacet_ind.shrink_to_fit();
        cofacets.shrink_to_fit();
    }

    // Construct cofacets from previously defined facets
    void construct_cofacets() {

        std::vector<std::vector<int> > cell_list(ncells, std::vector<int>());
        for(int i = 0; i < ncells; i++) {
            auto range = get_facet_range(i);
            for(auto j = range.first; j != range.second; j++) {
                cell_list[*j].push_back(i);
            }
        }

        cofacet_ind.clear();
        cofacets.clear();

        cofacet_ind.push_back(0);
        for(int i = 0; i < ncells; i++) {
            cofacet_ind.push_back(cofacet_ind[i] + cell_list[i].size());
            cofacets.insert(cofacets.end(), cell_list[i].begin(), cell_list[i].end());
        }

    }

};




//////////////////////////////////////////////////////////////////////////
// Cell complex modification
//////////////////////////////////////////////////////////////////////////





std::tuple<CellComplex, std::unordered_map<int, int> >  prune_cell_complex_map(std::vector<int> &rem_cells, CellComplex &comp) {

    std::unordered_set<int> full_rem_set;

    for(int c: rem_cells) {
        auto cofaces = comp.get_cofaces(c);
        full_rem_set.insert(cofaces.begin(), cofaces.end());
    }

    CellComplex red_comp(comp.dim, comp.regular, comp.oriented);

    std::unordered_map<int, int> full_to_reduced;

    for(int c = 0; c < comp.ncells; c++) {

        if(full_rem_set.count(c)) {
            continue;
        }


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

        full_to_reduced[c] = red_comp.ncells;

        red_comp.add_cell(red_comp.ndcells[comp.get_dim(c)], comp.get_dim(c), facets, coeffs);




    }

    red_comp.construct_cofacets();
    red_comp.make_compressed();

    return std::make_tuple(red_comp, full_to_reduced);


}

CellComplex prune_cell_complex(std::vector<int> &rem_cells, CellComplex &comp) {

    auto result = prune_cell_complex_map(rem_cells, comp);
    return std::get<0>(result);

}




bool check_boundary_op(CellComplex &comp) {
    bool valid = true;

    for(int i = 0; i < comp.ncells; i++) {
        if(comp.get_dim(i) > 0) {
            if(!comp.regular || comp.oriented) {
                std::unordered_map<int, int> sub_face_coeffs;

                std::unordered_map<int, int> coeffsi = comp.get_coeffs(i);
                for(auto j: coeffsi) {
                    std::unordered_map<int, int> coeffsj = comp.get_coeffs(j.first);
                    for(auto k: coeffsj) {
                        sub_face_coeffs[k.first] += coeffsi[j.first] * coeffsj[k.first];
                    }

                }

                for(auto j: sub_face_coeffs) {
                    if((comp.oriented && j.second != 0) || (!comp.oriented && j.second % 2 != 0) ) {
                        py::print("Error:", i, j.first, j.second);
                        valid = false;
                    }
                }


            } else {
                std::unordered_set<int> sub_faces;

                auto rangei = comp.get_facet_range(i);

                for(auto iti = rangei.first; iti != rangei.second; iti++) {

                    auto rangej = comp.get_facet_range(*iti);

                    for(auto itj = rangej.first; itj != rangej.second; itj++) {
                        if(sub_faces.count(*itj)) {
                            sub_faces.erase(*itj);
                        } else {
                            sub_faces.insert(*itj);
                        }
                    }

                }


                if(!sub_faces.empty()) {
                    py::print("Error:", i);
                    valid = false;

                }


            }

        }
    }

    return valid;

}


// std::unordered_set<int> get_boundary(std::unordered_set<int> &cells, CellComplex &comp) {
//     std::unordered_set<int> cycle;
//     if(comp.regular) {
//         for(auto c: cells) {
//             auto range = comp.get_facet_range(c);
//             for(auto it = range.first; it != range.second; it++) {
//                 int a = *it;
//                 if(cycle.count(a)) {
//                     cycle.erase(a);
//                 } else {
//                     cycle.insert(a);
//                 }
//             }
//         }
//     }

//     return cycle;
// }



#endif // CELLCOMPLEX_HPP
