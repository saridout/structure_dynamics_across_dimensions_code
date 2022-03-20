#include <iostream>
#include <fstream>
#include "State/StaticState.h"
#include "Potentials/Potentials.h"
//#include "Potentials/RegisterPotentials.h"
#include "Boundaries/Boxes.h"
#include "Resources/Exception.h"
#include "Computers/StaticComputer.h"
#include "Computers/MatrixInterface.h"
#include "Minimization/minimizer.h"
#include "Database/Database.h"
#include "Resources/srResources.h"
//#include "Resources/SimpleGrid.h"
#include "Resources/Helper.h"
#include <algorithm>
#include <iterator>
using namespace LiuJamming;

//#define DIM 2
using namespace std;

int main(int argc, char* argv[])
{
    //Disable harsh errors.
    NcError nc_err(NcError::silent_nonfatal);

    //the number of particles

    std::string dir = argv[1];
    int number = atoi(argv[2]);
    int Mmin = atoi(argv[3]);
    int Mmax = atoi(argv[4]);
    std::string filename = dir + "/state.nc";
    std::string framesfile = dir+"/drops.txt";
    int frame;
    std::vector<int> frames;
    std::ifstream stream(framesfile);
    double temp, trash;

    while (stream >> temp >> trash) { frames.push_back(int(temp));}
    CStaticState<DIM> System(number);
    Mmax = min(Mmax,int(frames.size()));
    CStaticDatabase<DIM> readdb(number,filename);
    for (int ff = Mmin; ff < Mmax; ff++) {
    cout << "ff: " << ff << "len(frames): " << frames.size() << std::endl << flush;

    frame = frames[ff];
    bool bad_mode = true;
    while (bad_mode) {
    readdb.Read(System,frame);

    //create a computer (an object that computes various properties of the system)
    CStaticComputer<DIM> Computer(System);

    //create a minimizer (an object that will minimize the system to its inherent structure)
    CSimpleMinimizer<DIM> Minimizer(Computer);

    Minimizer.minimizeFIRE();
    Computer.StdPrepareSystem();
    Computer.CalculateStdData();
    double eigenvectors[DIM*number];
    Eigen::VectorXd Eigenvalues;
    int numberRattler = Computer.RattlerMap.size();
    std::string outfilename = dir+"/lowmodesEnth/" +i_to_string(frame)+".txt";

    ofstream outputFile("blah.txt");
    double d2min[number];

    std::ifstream input(outfilename);
    //if we can't open, diagonalize!
    if (input.fail()) {

    //Because PV is linear, it drops out of hessian
    //hessian for enthalpy is therefore same as "generalize hessian" for energy
    //which Carl already has code to compute (see PRE anisotropies etc. paper maybe?)
    Eigen::Matrix<dbl,DIM,DIM> compress = Eigen::Matrix<dbl,DIM,DIM>::Identity() / DIM;
    cout << "blah\n" << flush;
    Computer.Data.H.A.resize(Computer.Data.H.A.innerSize()+1,Computer.Data.H.A.innerSize()+1);
    Computer.Bonds.ComputeGeneralizedHessian(Computer.Data.H.A,compress);
    cout << "blah\n" << flush;
    Computer.Data.H.Diagonalize(100);

    cout << "lowest 10 eigenvalues: ";
    for(int i = 0 ; i < 10 ; i++)
        cout << Computer.Data.H.Eigenvalues[i] << "\t";
    cout << endl;
    outputFile.close();
    outputFile.open(outfilename);

    outputFile << std::setprecision(9);

    double norm = 0;
    int numberRattler = Computer.RattlerMap.size();

    for(int i = 0 ; i < DIM*number ; i++)
        eigenvectors[i] = 0.0;

    for(int i = 0 ; i < numberRattler ; i++)
        for(int d = 0 ; d < DIM ; d++) {
            eigenvectors[DIM*Computer.RattlerMap[i]+d] = Computer.Data.H.Eigenvectors[DIM*(numberRattler*DIM+1) + DIM*i+d];
            norm += POW2(Computer.Data.H.Eigenvectors[DIM*numberRattler*DIM + DIM*i+d]);
        }
    norm = sqrt(norm);

    outputFile << Computer.Data.H.Eigenvalues[DIM] << "\n";
    for(int i = 0 ; i < DIM*number+1 ; i++) {
        eigenvectors[i] /= norm;
        outputFile << eigenvectors[i] << "\n";
    }
    outputFile << endl;
    Eigenvalues = Eigen::VectorXd::Zero(100);
    Eigenvalues(DIM) = Computer.Data.H.Eigenvalues[DIM]; 
    

    }
    else {
        cout << "mode has already been computed, we read the existing file..." << std::endl;
        cout << outfilename << std::endl;
        dbl current_number = 0;
        int iin = 0;
        Eigenvalues = Eigen::VectorXd::Zero(100);
        input >> current_number;
        Eigenvalues(DIM) = current_number;
          
        while (input >> current_number){
           eigenvectors[iin] = current_number;
           iin++;
        }
        input.close();

    }

    //compute d2min of the eigenvectors
    list<list<CSimpleNeighbor<DIM> > > neighborList;
    list<int> particles;
    for(int i = 0 ; i < numberRattler ; i++)
        particles.push_back(Computer.RattlerMap[i]);

    constructNeighborList(System, particles, 1.5, neighborList);
    cout << neighborList.size() << "\n";
    for(int i = 0 ; i < number ; i++)
        d2min[i] = 0;

    double DispSqSum = 0.0;
    Eigen::Matrix<double, DIM, DIM> X;
    Eigen::Matrix<double, DIM, DIM> Y;
    Eigen::Matrix<double, DIM, DIM> Transform;
    Eigen::Map<Eigen::VectorXd> evec(eigenvectors, DIM*number);

    list<list<CSimpleNeighbor<DIM> > >::iterator it_i;
    for(it_i = neighborList.begin() ; it_i != neighborList.end() ; it_i++) {
        list<CSimpleNeighbor<DIM> >::iterator it_j;

        double D2Min = 0.0;
        double Neighbors = 0.0;

        X = Eigen::Array<double, DIM, DIM>::Zero();
        Y = Eigen::Array<double, DIM, DIM>::Zero();
        Transform = Eigen::Array<double, DIM, DIM>::Zero();
        DispSqSum = 0.0;
        int p_index = -1;
        int n_index = -1;
        for(it_j = (*it_i).begin() ; it_j != (*it_i).end() ; it_j++) {
            p_index = (*it_j).i;
            n_index = (*it_j).j;
            if (Computer.RattlerMap.inv(n_index) > -1 && Computer.RattlerMap.inv(p_index) > -1) {
            Eigen::Matrix<double, DIM, 1> dvec_I = (*it_j).Displacement;
            Eigen::Matrix<double, DIM, 1> dvec_J = dvec_I + evec.segment<DIM>(DIM*(*it_j).i)
                                                          - evec.segment<DIM>(DIM*(*it_j).j);

            X += dvec_J*dvec_I.transpose();
            Y += dvec_I*dvec_I.transpose();
            DispSqSum += dvec_J.dot(dvec_J);

            Neighbors++;
            }
        }

        Transform = X*Y.inverse();
        d2min[p_index] = (DispSqSum - 2.0*(Transform*X.transpose()).trace() +
                         (Transform * Y * Transform.transpose()).trace())/Neighbors;

    }
    cout << "d2min compute\n"<<flush;
    outfilename = dir+"/moded2minEnth/" +i_to_string(frame)+".txt";
    outputFile.close();
    outputFile.open(outfilename);

    outputFile << std::setprecision(9);

    for(int i = 0 ; i < number ; i++)
        outputFile << d2min[i] << "\n";

    outputFile << endl;

    outputFile.close();
        
    frame -= 1;
    bad_mode = Eigenvalues(DIM) < 0;
    std::sort(&d2min[0],&d2min[number]);

    dbl d2min_mode = d2min[(number/2)];

    cout << "d2min_mode:" << d2min_mode << "\n";
    bad_mode = bad_mode || (d2min_mode < 1e-12); //translational mode
    } 
    }
    return 0;

}
