
#ifndef HELPER
#define HELPER

#include <iostream>
#include "State/StaticState.h"
#include "SimpleGrid.h"

using namespace LiuJamming;
using namespace std;
//constructs a neighbor list. should probably switch to a grid system.
void constructNeighborList(const CStaticState<DIM> &state,
                           const list<int> &particles, double cutoff,
                           list<list<CSimpleNeighbor<DIM> > > &neighborList)
{
    double cutoffsq = cutoff*cutoff;
    neighborList.clear();

    Eigen::Matrix<double,DIM,1> displacement;

    for(list<int>::const_iterator it = particles.begin() ;
        it != particles.end() ; it++)
    {
        list<CSimpleNeighbor<DIM> > neighbors;

        for(int j = 0 ; j < state.GetParticleNumber() ; j++)
        {
            if( j != (*it))
            {
                state.GetDisplacement((*it),j,displacement);
                if(displacement.squaredNorm() < cutoffsq)
                        neighbors.push_back(CSimpleNeighbor<DIM>((*it),j,
                                                                 displacement));
            }
        }
        neighborList.push_back(neighbors);
    }

}


#if DIM == 2
Eigen::MatrixXd simple_anguloradial_bin(const CStaticState<DIM> &state, const list<list<CSimpleNeighbor<DIM> > > &neighborList, std::vector<int> classes, double rmin, double rmax, int nr, int nt) {
    int n_classes = *std::max_element(classes.begin(),classes.end()) + 1;
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(state.GetN(),nr*nt*n_classes*n_classes);
    cout << "output length: " << output.innerSize()  << " " << output.outerSize() << "\n";
    for(auto it_i = neighborList.begin() ; it_i != neighborList.end() ; it_i++) {
        for(auto it_j = (*it_i).begin() ; it_j != (*it_i).end() ; it_j++) {
            
           Eigen::Matrix<dbl,DIM,1> r = (*it_j).Displacement;
           double rlen = sqrt(r.dot(r));
           Eigen::Matrix<dbl,DIM,1> a = {1.0/sqrt(2.0), 1.0/sqrt(2.0)};
           double t = acos(abs(r.dot(a))/rlen);
             
           int rbin = nr*(rlen - rmin)/(rmax-rmin);
           int tbin = nt*t*2.0/3.1415927;
           output((*it_j).i, (n_classes*classes[(*it_j).i] + classes[(*it_j).j]) * nr * nt + rbin*nt + tbin) += 1.0;
        }
    }
    return output;
}
#endif
#endif
