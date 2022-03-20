#ifndef LANDSCAPE_HPP
#define LANDSCAPE_HPP
    
#include "eigen_macros.hpp"
#include <vector>
#include <utility>
#include <algorithm>
#include "math.h"
    
    
#include <pybind11/pybind11.h>
namespace py = pybind11;


// Persistence landscape algorithms taken from:
// Bubenik, P., & Dłotko, P. (2017). A persistence landscapes toolbox for topological statistics. Journal of Symbolic Computation, 78, 91–114. https://doi.org/10.1016/j.jsc.2016.03.009



// Construct persistence landscape from birth-death pairs
// Persistence landscape consists of k pairs of lists of critical point coordinates (tips of tent functions)
// Only works for one dimension at a time
std::vector< std::pair<std::vector<double>, std::vector<double> > > construct_landscape(RXVec birth, RXVec death) {
    
    auto cmp = [](const std::pair<double, double> &lhs, const std::pair<double, double> &rhs) {
        
        
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        } else {
            return lhs.second > rhs.second;
        }
        
    };
    
    
    std::vector<std::pair<double, double> > pairs;
    // Add all birth-death pairs to list
    for(int i = 0; i < birth.size(); i++) {
        pairs.emplace_back(birth(i), death(i));
    }
    
    std::sort(pairs.begin(), pairs.end(), cmp);
            
    std::vector< std::pair<std::vector<double>, std::vector<double> > > landscape;
    int k = 0;
    while(!pairs.empty()) {
        
        // Initialize kth landscape
        landscape.emplace_back();
        
        // Set interator to first element
        auto itp = pairs.begin();
        
        // Start by popping first pair
        auto pair = *itp;
        itp = pairs.erase(itp);
        
        double b = pair.first;
        double d = pair.second;
        
        // Insert first two crtical points
        landscape[k].first.push_back(b);
        landscape[k].second.push_back(0.0);
                
        landscape[k].first.push_back((b+d)/2.0);
        landscape[k].second.push_back((d-b)/2.0);
                
        
        // Construct rest of kth envelope
        while(true) {
            
            // If at end of A, then finish envelope
            if(itp == pairs.end()) {                
                landscape[k].first.push_back(d);
                landscape[k].second.push_back(0.0);
                                
                break;
            }
            
            // Iterate through elements in A starting at itp and find d' > d if exists
            std::vector<std::pair<double, double> >::iterator itprime = itp;
            bool is_max = true;
            for(auto it = itp; it != pairs.end(); it++) {
                           
                // If d' > d, then stop
                if((*it).second > d) {
                    itprime = it;
                    is_max = false;
                    break;
                }
            }
            
            
             // If d > d', then finish envolope
            if(is_max) {
                landscape[k].first.push_back(d);
                landscape[k].second.push_back(0.0);

                break;
            }
            
            
            auto prime = *itprime;
                        
            double bp = prime.first;
            double dp = prime.second;
            
            // Pop (b', d') and move pointer to next position
            itp = pairs.erase(itprime);
            
            if(bp > d) {
                landscape[k].first.push_back(d);
                landscape[k].second.push_back(0.0);
            }
            
            if(bp >= d) {
                landscape[k].first.push_back(bp);
                landscape[k].second.push_back(0.0);;
            } else {
                landscape[k].first.push_back((bp+d)/2.0);
                landscape[k].second.push_back((d-bp)/2.0);
                
                // Add (bp, d) to list of pairs in sorted positions, but located after pointer                
                auto it = std::upper_bound(itp, pairs.end(), std::make_pair(bp, d));
                itp = pairs.insert(it, std::make_pair(bp, d));
                // Move pointer to pair directly after the newly inserter pair
                itp++;
                
            }
            
            landscape[k].first.push_back((bp+dp)/2.0);
            landscape[k].second.push_back((dp-bp)/2.0);
                        
            b = bp;
            d = dp;
            
            
        }
        
        // Increment envelope
        k++;
        
    }
    
    return landscape;
    
}

// Evaluate single envelope of landscape at x coord and return y coord
double eval_landscape(double x, std::vector<double> &X, std::vector<double> &Y) {
    
    if(X.empty() || x <= X.front() || x >= X.back()) {
        return 0.0;
    }
    
    // Pointer to first element greater than x
    auto up = std::upper_bound(X.begin(), X.end(), x);
    
    // Index of upper bound
    int i = up - X.begin();
    
    double x0 = X[i-1];
    double y0 = Y[i-1];
    
    double x1 = X[i];
    double y1 = Y[i];
    
    // If two points are almost on top of one another
    // This prevents divide by zero errors
    if (x1 - x0 < 1e-12) {
        return y0;
    }
    
    return (y0*(x1-x) + y1*(x-x0)) / (x1-x0);
    
}

// Create linear combination of persistence landscapes
// Takes in list of landscapes and vector of coefficients
std::vector< std::pair<std::vector<double>, std::vector<double> > > combine_landscapes(std::vector< std::vector< std::pair<std::vector<double>, std::vector<double> > > > &landscapes, RXVec a) {
    
    

    // List of landscapes present for each value of k
    std::vector<std::vector<int> > present;
    for(std::size_t i = 0; i < landscapes.size(); i++) {
        if(landscapes[i].size() > present.size()) {
            present.resize(landscapes[i].size());
        }
        
        for(std::size_t k = 0; k < landscapes[i].size(); k++) {
            present[k].push_back(i);
        }
    }
    
    
    
    std::vector< std::pair<std::vector<double>, std::vector<double> > > combo(present.size());
    
    
    // Calculate linear combination for each envelope
    for(std::size_t k = 0; k < present.size(); k++) {
        
        // Collect all x coords
        for(int i: present[k]) {
            combo[k].first.insert(combo[k].first.end(), landscapes[i][k].first.begin(), landscapes[i][k].first.end());
        }
        
        // Sort x coords
        std::sort(combo[k].first.begin(), combo[k].first.end());
        
        auto cmp = [](const double lhs, const double rhs) {
            return rhs - lhs < 1e-12;
        };
        
        // Remove duplicates
        combo[k].first.erase(std::unique(combo[k].first.begin(), combo[k].first.end(), cmp), combo[k].first.end());
                
        // Construct Yk
        for(double Xk: combo[k].first) {
            
            double Yk = 0.0;
            for(int i: present[k]) {
                Yk += a(i) *  eval_landscape(Xk, landscapes[i][k].first, landscapes[i][k].second);
            }
            
            combo[k].second.push_back(Yk);
                
        }
        
        
        
    }
    
    
    
    return combo;
    
}

// Add landscapes a1*l1 + a2*l2
std::vector< std::pair<std::vector<double>, std::vector<double> > > combine_landscapes(
                            std::vector< std::pair<std::vector<double>, std::vector<double> > > &landscape1, 
                            std::vector< std::pair<std::vector<double>, std::vector<double> > > &landscape2, 
                            double a1, double a2) {
    
    std::size_t Nk = std::max(landscape1.size(), landscape2.size());
    
    
    std::vector< std::pair<std::vector<double>, std::vector<double> > > combo(Nk);
    
    
    // Calculate linear combination for each envelope
    for(std::size_t k = 0; k < Nk; k++) {
        
        // Collect all x coords
        if(landscape1.size() > k) {
            combo[k].first.insert(combo[k].first.end(), landscape1[k].first.begin(), landscape1[k].first.end());
        }
        
        if(landscape2.size() > k) {
            combo[k].first.insert(combo[k].first.end(), landscape2[k].first.begin(), landscape2[k].first.end());
        }
        
        // Sort x coords
        std::sort(combo[k].first.begin(), combo[k].first.end());
        
        auto cmp = [](const double lhs, const double rhs) {
            return rhs - lhs < 1e-12;
        };
        
        // Remove duplicates
        combo[k].first.erase(std::unique(combo[k].first.begin(), combo[k].first.end(), cmp), combo[k].first.end());
                
        // Construct Yk
        for(double Xk: combo[k].first) {
            
            double Yk = 0.0;
            
            if(landscape1.size() > k) {
                Yk += a1 *  eval_landscape(Xk, landscape1[k].first, landscape1[k].second);
            }
            
            if(landscape2.size() > k) {
                Yk += a2 *  eval_landscape(Xk, landscape2[k].first, landscape2[k].second);
            }
            
            combo[k].second.push_back(Yk);
                
        }
        
        
        
    }
    
    
    
    return combo;
    
}


std::vector< std::pair<std::vector<double>, std::vector<double> > > calc_average_landscape(std::vector< std::vector< std::pair<std::vector<double>, std::vector<double> > > > &landscapes) {
    
    int N = landscapes.size();
    XVec a = XVec::Constant(N, 1.0/N);
    
    return combine_landscapes(landscapes, a);
    
}


// Calculate norm of persistence landscape
// sum up \int dx (ax+b)^2 = 1/3 (x1-x0)(y0^2+y0y1+y1^2)
double calc_landscape_norm(std::vector< std::pair<std::vector<double>, std::vector<double> > > &landscape) {
    
    double norm = 0.0;
    for(std::size_t k = 0; k < landscape.size(); k++) {
        int N = landscape[k].first.size();
        if(N <= 1) {
            continue;
        }
        
        XArrMap X(landscape[k].first.data(), N);
        XArrMap Y(landscape[k].second.data(), N);

        norm +=  (1.0/3.0 * (X.tail(N-1) - X.head(N-1)) 
            * (Y.head(N-1).pow(2.0) + Y.head(N-1)*Y.tail(N-1) + Y.tail(N-1).pow(2.0))).sum();
        
        
        
    }
        
    return sqrt(norm);
    
    
}

// Calculate norm of difference of pair of landscapes
double calc_landscape_dist(std::vector< std::pair<std::vector<double>, std::vector<double> > > &landscape1, 
                           std::vector< std::pair<std::vector<double>, std::vector<double> > > &landscape2) {

    
    auto diff = combine_landscapes(landscape1, landscape2, 1.0, -1.0);
    
    return calc_landscape_norm(diff);
    
}


XMat calc_dist_mat(std::vector< std::vector< std::pair<std::vector<double>, std::vector<double> > > > &landscapes) {

    XMat dists = XMat::Zero(landscapes.size(), landscapes.size());
    
    for(std::size_t i = 0; i < landscapes.size(); i++) {
        for(std::size_t j = i; j < landscapes.size(); j++) {
            
            dists(i, j) = calc_landscape_dist(landscapes[i], landscapes[j]);
                        
            dists(j, i) = dists(i, j);
        
        }
    }
    
    return dists;
    
}

XMat calc_dist_mat_norms(std::vector< std::vector< std::pair<std::vector<double>, std::vector<double> > > > &landscapes) {

    XMat norms = XMat::Zero(landscapes.size(), landscapes.size());
    
    for(std::size_t i = 0; i < landscapes.size(); i++) {
        for(std::size_t j = i; j < landscapes.size(); j++) {
            
            auto sum = combine_landscapes(landscapes[i], landscapes[j], 1.0, 1.0);
            
            norms(i, j) = calc_landscape_norm(sum);
                        
            norms(j, i) = norms(i, j);
        
        }
    }
    
    return norms;
    
}





// XMat calc_landscape(CRXVec birth, CRXVec death, CRXVec t, int K) {

//     int NL = (K > 0) ? K : birth.size();
    
//     XMat landscape = XMat::Zero(NL, t.size());
    
//     XVec b = birth.cwiseMin(death);
//     XVec d = birth.cwiseMax(death);
    
//     for(int i = 0; i < t.size(); i++) {

//         // \lambda_k(t) = max(0, min(t - b, d -t) )
        
//         XVec lambdakt = ((t[i] - b.array()).min(d.array() - t[i])).max(0.0);
        
//         std::sort(lambdakt.data(), lambdakt.data()+lambdakt.size(), std::greater<double>());
        
        
//         landscape.block(0, i, std::min(NL, (int)lambdakt.size()), 1) = lambdakt.segment(0, std::min(NL, (int)lambdakt.size()));
        
//     }
    
    
//     return landscape;
    
    
    
// }

// double calc_landscape_norm(CRXMat landscape, CRXVec t) {
    
//     XMat landscape2 = landscape.array().square();
    
//     XMat dland = landscape2.block(0, 1, landscape2.rows(), landscape2.cols()-1) + landscape2.block(0, 0, landscape2.rows(), landscape2.cols()-1);
    
//     XVec dt = t.segment(1, t.size()-1) - t.segment(0, t.size()-1);
    
//     return sqrt(0.5 * (dland * dt).sum());
    
// }


// double calc_landscape_dist(CRXMat landscapei, CRXMat landscapej, CRXVec t) {
        
//     XMat landscape2 = (landscapej - landscapei).array().square();
    
//     XMat dland = landscape2.block(0, 1, landscape2.rows(), landscape2.cols()-1) + landscape2.block(0, 0, landscape2.rows(), landscape2.cols()-1);
    
//     XVec dt = t.segment(1, t.size()-1) - t.segment(0, t.size()-1);
    
//     return sqrt(0.5 * (dland * dt).sum());
    
// }


// double calc_landscape_norm(CRXVec birth, CRXVec death, CRXVec t) {
    
//     XMat landscape2 = calc_landscape(birth, death, t, -1).array().square();
    
//     XMat dland = landscape2.block(0, 1, landscape2.rows(), landscape2.cols()-1) + landscape2.block(0, 0, landscape2.rows(), landscape2.cols()-1);
    
//     XVec dt = t.segment(1, t.size()-1) - t.segment(0, t.size()-1);
    
//     return sqrt(0.5 * (dland * dt).sum());
    
// }


// double calc_landscape_dist(CRXVec birth1, CRXVec death1, CRXVec birth2, CRXVec death2, CRXVec t) {
    
//     int K = std::max(birth1.size(), birth2.size());
    
//     XMat landscape2 = (calc_landscape(birth2, death2, t, K) - calc_landscape(birth1, death1, t, K)).array().square();
    
//     XMat dland = landscape2.block(0, 1, landscape2.rows(), landscape2.cols()-1) + landscape2.block(0, 0, landscape2.rows(), landscape2.cols()-1);
    
//     XVec dt = t.segment(1, t.size()-1) - t.segment(0, t.size()-1);
    
//     return sqrt(0.5 * (dland * dt).sum());
    
// }



#endif // LANDSCAPE_HPP
