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
    cout << "about to read from drops file\n" << flush;    
    while (stream >> temp >> trash) { frames.push_back(int(temp));}
    CStaticState<DIM> System(number);
    Mmax = min(Mmax,int(frames.size()));
    cout << "about to open readdb\n" << flush;
    CStaticDatabase<DIM> readdb(number,filename);
    cout << "did that\n" << flush;
    for (int ff = Mmin; ff < Mmax; ff++) {
    cout << "ff: " << ff << "len(frames): " << frames.size() << std::endl << flush;
    frame = frames[ff]; 
    bool bad_mode = true;
    while (bad_mode) {
    cout << "try to read frame " << frame << " out of " << readdb.GetNumRecs() << std::endl << flush;
    readdb.Read(System,frame);
    cout <<"...and of course that didn't fail, right?" << std::endl << flush;
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
   
    std::string outfilename = dir+"/lowmodes2/" +i_to_string(frame)+".txt";

    std::ofstream outputFile("blah.txt");
    std::ifstream input(outfilename);
    cout << "input load " << outfilename << std::endl << flush;
    double d2min[number];
    //if we can't open, diagonalize!
    if (input.fail()) {
        cerr << "Error: " << strerror(errno);
        input.close();
        Computer.Data.H.Diagonalize(100);

        cout << "lowest 10 eigenvalues: ";
        for(int i = 0 ; i < 10 ; i++)
            cout << Computer.Data.H.Eigenvalues[i] << "\t";
        cout << endl;
        outputFile.close();
        outputFile.open(outfilename);
        outputFile << std::setprecision(9);

        for(int i = 0 ; i < DIM*number ; i++)
            eigenvectors[i] = 0.0;

        for(int i = 0 ; i < numberRattler ; i++)
            for(int d = 0 ; d < DIM ; d++)
                eigenvectors[DIM*Computer.RattlerMap[i]+d] = Computer.Data.H.Eigenvectors[DIM*numberRattler*DIM + DIM*i+d];
        cout << " now write eigenvalue to file \n"; 
        outputFile << Computer.Data.H.Eigenvalues[DIM] << "\n";
        for(int i = 0 ; i < DIM*number ; i++)
            outputFile << eigenvectors[i] << "\n";
        outputFile << endl;

        //compute d2min of the eigenvectors
        list<list<CSimpleNeighbor<DIM> > > neighborList;
        list<int> particles;
        for(int i = 0 ; i < number ; i++)
            particles.push_back(i);

        constructNeighborList(System, particles, 1.5, neighborList);

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

            for(it_j = (*it_i).begin() ; it_j != (*it_i).end() ; it_j++) {
                p_index = (*it_j).i;

                Eigen::Matrix<double, DIM, 1> dvec_I = (*it_j).Displacement;
                Eigen::Matrix<double, DIM, 1> dvec_J = dvec_I + evec.segment<DIM>(DIM*(*it_j).i)
                                                          - evec.segment<DIM>(DIM*(*it_j).j);

                X += dvec_J*dvec_I.transpose();
                Y += dvec_I*dvec_I.transpose();
                DispSqSum += dvec_J.dot(dvec_J);

                Neighbors++;
            }

            Transform = X*Y.inverse();
            d2min[p_index] = (DispSqSum - 2.0*(Transform*X.transpose()).trace() +
                         (Transform * Y * Transform.transpose()).trace())/Neighbors;

        }
        outfilename = dir+"/moded2min2/" +i_to_string(frame)+".txt";
        outputFile.close();
        outputFile.open(outfilename);

        outputFile << std::setprecision(9);

        for(int i = 0 ; i < number ; i++)
            outputFile << d2min[i] << "\n";

        outputFile << endl;

        outputFile.close();
        Eigenvalues = Eigen::VectorXd::Zero(100);
        Eigenvalues(DIM) = Computer.Data.H.Eigenvalues[DIM]; 
    }
    else { //reading file 
        cout << "mode has already been computed, we read the existing file..." << std::endl;
        dbl current_number = 0;
        int iin = 0;
        Eigenvalues = Eigen::VectorXd::Zero(100);
        input >> current_number;
        Eigenvalues(DIM) = current_number;
        //loaded Eigenvalues, now let's do d2min
        input.close();
        input.open(dir+"/moded2min2/" +i_to_string(frame)+".txt");
        iin = 0; 
        while (input >> current_number){
            d2min[iin] = current_number;
            iin++;
        }
        input.close();
    }
    frame -= 1;
    bad_mode = Eigenvalues(DIM) < 0;
    std::sort(&d2min[0],&d2min[number]);
    dbl d2min_mode = d2min[(number/2)];
    bad_mode = bad_mode || (d2min_mode < 1e-13); //translational mode 
    /*cout << d2min_mode << " " << bad_mode << "\n";
    for (int j = 0; j < 12; j++) {
    for (int i = 0; i < DIM + 1; i++) { 
      cout << Computer.Data.H.Eigenvectors[i*numberRattler*DIM + j] << " ";
    }
    cout << "\n"; }
    cout << frames[Mmin] << " " << frame << "\n";*/
    }
    }
    return 0;
    
}
