/*
* Shear packing, slowing down as we approach any drop in stress above a threshold.
* Apply "remembered" linear response before minimizing during elastic branch.
* As of April 9,2019, use affine initial displacement near plastic event for numerical stability.
* In future: possibly replace this with a more stable numerical derivative calculation.
* Do not calculate elastic moduli & hessian, to reduce memory usage
* Preemptable: can stop and start at will.
* Sean Ridout
*/
#include <iostream>
#include "State/StaticState.h"
#include "Boundaries/Boxes.h"
#include "Resources/Exception.h"
#include "Computers/StaticComputer.h"
#include "Computers/MatrixInterface.h"
#include "Minimization/minimizer.h"
#include "Database/Database.h"
#include <sstream>
#include "Resources/srResources.h"
#include "dirs.h"
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netcdfcpp.h>
#include "Protocols/MemoryStrain.h"
using namespace LiuJamming;
//#define DIM 2
typedef Eigen::Matrix<dbl, DIM, DIM> dmat;
using namespace std;

int main(int argc, char *argv[])
{
	dbl tol = 1e-12;
	dbl phi = 0.95;
  string phi_string = "0.95";
	int N = 4096;
	string N_string;
	int seed = 0;
	string seed_string;
	string maxstrain;
	dbl strain_step = 1e-5;
	int rP = 1;
	std::string rP_string = "1";
  int steps = 0;
        int max_events = 999999;
	int in;
        bool test = false;
        while((in = getopt(argc,argv,"n:p:s:m:e:g:r:x:t::")) != -1)
                switch(in)
                {
                    case 'n': N_string = optarg; N = atoi(N_string.c_str());cout << "n " <<flush; break;
                    case 'p': phi_string = optarg; phi = atof(phi_string.c_str()); cout << "p" << flush;  break;
      							case 'e': tol = atof(optarg); break;
										case 'm': steps = atoi(optarg); break;
										case 's': seed_string = optarg; seed = atoi(optarg); break;
                    case 'g': strain_step = atof(optarg); break;
										case 'r': rP_string = optarg; rP = atoi(optarg); break;
                   case 'x': max_events = atoi(optarg); break;
                   case 't': test = true; break;
	                default :
                                abort();
                }



	NcError err(NcError::silent_nonfatal);
  std::string dir = (std::string)SAR_STATE_DIR + "/memsshear/"+i_to_string(DIM)+"d/";
	//string dir = (std::string)SAR_STATE_DIR + (std::string)"sshear/";
	std::string datafilename = dir+i_to_string(DIM)+"dshear_N="+N_string+"_phi="+phi_string+"_dγ="+d_to_string(strain_step)+"_rP="+rP_string+"_seed="+seed_string+"/data.nc";
	std::string statefilename = dir+ i_to_string(DIM) + "dshear_N="+N_string+"_phi="+phi_string+"_dγ="+d_to_string(strain_step)+"_rP="+rP_string+"_seed="+seed_string+"/state.nc";
	//first determine whether or not the output files already exist, indicating existing progress.
	CStaticState<DIM> s(N);
	int depth= 0;
	int maxdepth = 10;
	int tempcount = 0;
	int existing_M = 0;
  NcFile::FileMode mode; 
	cout << "statefilename: " << statefilename << "\n" << flush;
	if (file_exists(statefilename)) {
		assert(file_exists(datafilename));
		//now check whether or not we got interrupted mid-write.
		//for now just abort, if it actually ends up happening we can handle this problem
		CStaticDatabase<DIM> sread(N,statefilename,NcFile::ReadOnly);
		existing_M = sread.GetNumRecs();
		CStdDataDatabase<DIM> dread(datafilename,NcFile::ReadOnly);
		assert(existing_M == dread.GetNumRecs());
		cout << "done initial reads\n" << flush;
		//now we're ready to read in the current state
		cout << "about to do first test read\n" << flush;
		sread.Read(s,0);
		cout << "done first test read\n" << flush;
		sread.Read(s,existing_M-2);
		cout <<"done test read\n" << flush;
		sread.Read(s,existing_M-1);
		cout << "done second read\n" << flush;
		//now infer depth
		dmat box;
		s.GetBox()->GetTransformation(box);
		cout << box << "\n\n";
		dbl old_shear = box(1,0) / box(0,0);
		sread.Read(s,existing_M-2);
		s.GetBox()->GetTransformation(box);
		cout << box << "\n\n";
		old_shear -= box(1,0) / box(0,0);
		depth = std::round(std::log( strain_step / old_shear) / std::log(5)) - 1;
		if (depth < 1) { depth = 1;} //accounts for recording period > 1

		if (depth > 1) { //infer tempcount
			int curr_depth = depth;
			while (curr_depth == depth) {
				tempcount += 1;
				s.GetBox()->GetTransformation(box);
				old_shear = box(1,0) / box(0,0);
				sread.Read(s,existing_M-2 - tempcount);
				s.GetBox()->GetTransformation(box);
				old_shear -= box(1,0) / box(0,0);
				curr_depth = std::round(strain_step / old_shear) - 1;
			}
		}
		cout <<"prev depth: " << depth << ", prev tempcount: " << tempcount << "\n" << flush;
		//now prepare for writing
    mode = NcFile::Write;

	}
	else { //trajectory in progress does not exist
    mode = NcFile::New;
	}

		CStdDataDatabase<DIM> dwrite(datafilename,mode);
		CStaticDatabase<DIM> swrite(N, statefilename,mode);
  cout << dir+ i_to_string(DIM) + "dshear_N="+N_string+"_phi="+phi_string+"_dγ="+d_to_string(strain_step)+"_rP="+rP_string+"_seed="+seed_string<< "\n";

	if (existing_M == 0) {
		s.RandomizePositions(seed);
		s.SetRadiiBi();
		s.SetPackingFraction(phi);
		s.SetPotentialHertzian();
                
		depth = 1;
	  tempcount = 0;
    
	}

	CStaticComputer<DIM> c(s);
	//initial Minimization
	CSimpleMinimizer<DIM> miner(c);
	miner.minimizeFIRE(tol);
  c.StdPrepareSystem();
	c.CalculateStdData(false, false);
        if (existing_M == 0) {
            swrite.Write(s);
            dwrite.Write(c.Data);

        }
	dmat InitTrans;
	s.GetBox()->GetTransformation(InitTrans);
	dmat strain_tensor = dmat::Zero();
	strain_tensor(1,0) = 1.0;


  int events = 0; //this is the wrong place to set it, but the max_events is meant to be very crude
  Protocols::DoubleMemoryStrain<DIM> mem(&s,&miner,strain_tensor);
  for (int i = existing_M; i < steps; i++) { 
    if (events < max_events) {
		CStaticState<DIM> oldstate = s; //copy
    dbl oldEnergy = c.Data.Stress(0,1);
  	//dmat strain = (strain_step / std::pow(5.0,depth - 1.0))*strain_tensor;
    dbl ds = (strain_step / std::pow(5.0,depth - 1.0));
    c.StdPrepareSystem();
    Eigen::VectorXd pos;
    s.GetPositions(pos);
    cout << std::setprecision(10);
    if (depth == 1) 
      mem.strain_step(ds);
    else {
      c.AffineShear(ds*strain_tensor);
      miner.minimizeFIRE(tol);
    }
    s.GetPositions(pos);
	  c.StdPrepareSystem();
  	c.CalculateStdData(false, false);
    cout << oldEnergy << " " << c.Data.Stress(0,1) << " " << depth <<  "\n" << flush;
		if (c.Data.Stress(0,1) < (1-0.01*sgn(oldEnergy))*oldEnergy) { //go slower
			if (depth < maxdepth) {
			  s = oldstate; //go back
        c.StdPrepareSystem();
        c.CalculateStdData(false, false);
      	depth += 1;
        tempcount = 0;
			}
			else {
        depth = 1;
        mem.forget();
    		dwrite.Write(c.Data);
		    swrite.Write(s);
        events += 1;
      } //resume normal speed
		}
    else {
      tempcount += 1;
			if (depth > 1 || (i % rP == 0)) { //don't record as much data in elastic loading
       if (!test){	dwrite.Write(c.Data);
	    	swrite.Write(s); }
			}
      if ((depth > 1) && (tempcount >= 5)) {
        tempcount = 0;
        depth = 1;
        mem.forget();
      }
    }

	} }
	//we will almost certainly run out of walltime, so NOTHING should happen after this loop!!!!!!!!!!!

}
