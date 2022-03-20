#ifndef H_STRAINPROTOCOLS
#define H_STRAINPROTOCOLS

#include "State/StaticState.h"
#include "Computers/StaticComputer.h"
#include "Minimization/minimizer.h"
#include "Resources/defs.h"
#include "Protocols/MemoryStrain.h"

//#include "Resources/Helper.h"
using namespace LiuJamming;
namespace Protocols {
/*
  int strainToStressDrop(CStaticState<DIM> & s,dmat strain_tensor, dbl ds, dbl ratio, int maxdepth, dbl tol=1e-12) {
    CStaticComputer<DIM> c(s);
	  CSimpleMinimizer<DIM> miner(c);
    dmat InitTrans;
	  s.GetBox()->GetTransformation(InitTrans);
    int depth = 1;
	  dbl gamma = 0;
    int tempcount = 0;
    dbl dratio = (dbl)ratio;
    int max_steps = 1000;
    for (int i = 0; i < max_steps; i++) {
  		CStaticState<DIM> oldstate = s; //copy
      dbl oldStress = (c.Data.Stress*strain_tensor).trace();
    	dmat strain = (ds / std::pow(dratio,depth - 1.0))*strain_tensor;
      c.StdPrepareSystem();
      Eigen::VectorXd pos;
  		c.ShearWithNonAffine(strain);
      miner.minimizeFIRE(tol);
  	  c.StdPrepareSystem();
    	c.CalculateStdData();
  		if ((c.Data.Stress(0,1)*strain_tensor).trace() < oldStress) { //go slower
  			if (depth < maxdepth) {
  			  s = oldstate; //go back
          gamma -= ds / std::pow(dratio,depth - 1.0);
          c.StdPrepareSystem();
          c.CalculateStdData();
        	depth += 1;
          tempcount = 0;
  			}
  			else { //we have gotten as close as we care to
          s = oldstate;
          return 0;
        }
  		}
      else {
        tempcount += 1;
        if (tempcount >= ratio) {
          tempcount = 0;
          depth = 1;
        }
      }
  	}
    return -1; //should never happen

  }

  int strainToInstability(CStaticState<DIM> & s,dmat strain_tensor, dbl ds, dbl ratio, int maxdepth, dbl tol=1e-12,bool NA = true,dbl minTimescale=-1.0,dbl eigThresh=1e-12) {
    CStaticComputer<DIM> c(s);
	  CSimpleMinimizer<DIM> miner(c);
    dmat InitTrans;
	  s.GetBox()->GetTransformation(InitTrans);
    int depth = 1;
	  dbl gamma = 0;
    int tempcount = 0;
    dbl dratio = (dbl)ratio;
    int max_steps = 1000;
    for (int i = 0; i < max_steps; i++) {
  		CStaticState<DIM> oldstate = s; //copy
      dbl oldStress = (c.Data.Stress*strain_tensor).trace();
    	dmat strain = (ds / std::pow(dratio,depth - 1.0))*strain_tensor;
      c.StdPrepareSystem();
      Eigen::VectorXd pos;
      if (NA) {c.ShearWithNonAffine(strain);}
      else {c.AffineShear(strain);}
      miner.minimizeFIRE(tol,-1,minTimescale);
  	  c.StdPrepareSystem();
    	c.CalculateStdData();
      //if depth > 1, look at eigenvalue too
      bool unstableEig = false;
      if (depth > 1) {
        c.Data.H.Diagonalize(100);
        unstableEig = c.Data.H.Eigenvalues[DIM] < eigThresh;
      }
  		if (((c.Data.Stress*strain_tensor).trace() < oldStress) || eigThresh) { //go slower
        cout << "GO BACK\n"<<flush;
  			if (depth < maxdepth) {
  			  s = oldstate; //go back
          gamma -= ds / std::pow(dratio,depth - 1.0);
          c.StdPrepareSystem();
          c.CalculateStdData();
        	depth += 1;
          tempcount = 0;
  			}
  			else { //we have gotten as close as we care to
          s = oldstate;
          return 0;
        }
  		}
      else {
        tempcount += 1;
        if (tempcount >= ratio) {
          tempcount = 0;
          depth = 1;
        }
      }
  	}
    return -1; //should never happen

  }


  dbl slowJamFromBelow(CStaticState<DIM> & s, dbl dphi, dbl ratio, int maxdepth, dbl tol=1e-12) {
    CStaticComputer<DIM> c(s);
	  CSimpleMinimizer<DIM> miner(c);
    dmat InitTrans;
    int depth = 1;
	  dbl gamma = 0;
    int tempcount = 0;
    dbl dratio = (dbl)ratio;
    int max_steps = 1000;
    Eigen::VectorXd philist = Eigen::VectorXd::Zero(max_steps);
    for (int i = 0; i < max_steps; i++) {
  		CStaticState<DIM> oldstate = s; //copy
      //dmat strain = -  (ds / std::pow(dratio,depth - 1.0))*dmat::Identity();
  	  //s.GetBox()->GetTransformation(InitTrans);
      //InitTrans *= dmat::Identity()+strain;
      //s.GetBox()->SetTransformation(InitTrans);
      dbl phi = s.GetPackingFraction();
      philist(i) = phi;
      s.SetPackingFraction(phi + dphi/std::pow(dratio,depth-1.0));
      miner.minimizeFIRE(tol);
  	  int jammed = 0;
      if(! c.StdPrepareSystem()) {
        c.CalculateStdData();
        jammed = c.Data.cijkl.CalculateBulkModulus() > 0.0001;
        cout << c.Data.cijkl.CalculateBulkModulus() << "\n";
      }
    	c.CalculateStdData();
  		if (jammed) { //go slower
  			if (depth < maxdepth) {
  			  s = oldstate; //go back
        	depth += 1;
          tempcount = 0;
  			}
  			else { //we have gotten as close as we care to
          s = oldstate;
          //for (int j = 0; j < i; j++) {cout << philist(j) << "\n";}
          return c.Data.Pressure;
        }
  		}
      else {
        tempcount += 1;
        if (tempcount >= ratio) {
          tempcount = 0;
          depth = 1;
        }
      }
  	}

    return -1; //should never happen


  }
*/
  //the default arguments push the limits of double precision for phi
  dbl slowCompressionToPtarg(CStaticState<DIM> & s, dbl Ptarg, dbl dphi=1e-5, dbl ratio=2, int maxdepth=34, dbl tol=1e-12) {
    CStaticComputer<DIM> c(s);
    CSimpleMinimizer<DIM> miner(c);
    dmat InitTrans;
    int depth = 1;
    dbl gamma = 0;
    int tempcount = 0;
    dbl dratio = (dbl)ratio;
    int max_steps = 1000;
    Eigen::VectorXd philist = Eigen::VectorXd::Zero(max_steps);
    for (int i = 0; i < max_steps; i++) {
      CStaticState<DIM> oldstate = s; //copy
      //dmat strain = -  (ds / std::pow(dratio,depth - 1.0))*dmat::Identity();
      //s.GetBox()->GetTransformation(InitTrans);
      //InitTrans *= dmat::Identity()+strain;
      //s.GetBox()->SetTransformation(InitTrans);
      dbl phi = s.GetPackingFraction();
      philist(i) = phi;
      s.SetPackingFraction(phi + dphi/std::pow(dratio,depth-1.0));
      miner.minimizeFIRE(tol);
      int jammed = 0;
      if(! c.StdPrepareSystem()) {
        c.CalculateStdData();
        jammed = c.Data.cijkl.CalculateBulkModulus() > 0.0001;
      }
      c.CalculateStdData();
      if (c.Data.Pressure > Ptarg) { //go slower
        if (depth < maxdepth) {
          s = oldstate; //go back
          depth += 1;
          tempcount = 0;
        }
        else { //we have gotten as close as we care to
          s = oldstate;
          //for (int j = 0; j < i; j++) {cout << philist(j) << "\n";}
          return c.Data.Pressure;
        }
      }
      else {
        tempcount += 1;
        if (tempcount >= ratio) {
          tempcount = 0;
          depth = 1;
        }
      }
    }

    return -1; //should never happen


  }


void slowMemoryPtarg(CStaticState<DIM> & s, dbl Ptarg, dbl P_rel_tol = 1e-2, dbl dstrain=1e-4, dbl ratio=2, dbl max_depth=32, dbl max_rel_change=0.1, dbl tol=1e-12)  {
    CStaticComputer<DIM> c(s);
    CSimpleMinimizer<DIM> miner(c);

    dmat strain_tensor = (1.0/DIM) * dmat::Identity();
        
    SingleMemoryStrain<DIM> mem(&s, &c,  &miner,strain_tensor);

    miner.minimizeFIRE(tol);
    c.StdPrepareSystem();
    c.CalculateStdData();

    int depth = 0;
    int tempcount = 0;
    

    while (std::abs(c.Data.Pressure - Ptarg) > Ptarg*P_rel_tol) {

        dbl prev_P = c.Data.Pressure;
        CStaticState<DIM> oldstate = s; //copy
        dbl ds = (dstrain / std::pow(ratio,depth))*sgn(c.Data.Pressure - Ptarg);
        std::cout << "Pressure: " << prev_P <<" depth: " << depth << " ds: " << ds << "\n"<<std::flush;
        if (depth >= 0)  
          mem.strain_step(ds);
        else {
          c.AffineShear(ds*strain_tensor);
          miner.minimizeFIRE(tol);
        }
        c.StdPrepareSystem();
  	c.CalculateStdData(false, false);

        if (std::abs(c.Data.Pressure - prev_P) > max_rel_change*prev_P) {
          if (depth < max_depth) {
            s = oldstate; //go back
            c.StdPrepareSystem();
            c.CalculateStdData(false, false);
            depth += 1;
            tempcount = 0;
			    }
          else {
            std::cout << "MAX DEPTH INSUFFICIENT?" << std::flush;
          }
          
        }
        else {
            tempcount += 1;

            if ((depth > 0) && (tempcount >= ratio)) {
              tempcount = 0;
              depth -= 1;
              //mem.forget();
            }
        }



    }

    

}
  /*
  dbl localYield(CStaticState<DIM> & s, int particle, dbl radius, dmat strain_tensor, dbl ds=1e-4, dbl ratio=5.0, int maxdepth=5, dbl tol=1e-12) {
    //make a computer and pin the particles outside the radius
    CStaticComputer<DIM> c(s);
    int N = s.GetN();
    vector<bool> FixedDof(DIM*N,true);
    for (int d = 0; d < DIM;d++) {FixedDof[particle*DIM+d] = false;}
    list<list<CSimpleNeighbor<DIM> > > neighborList;
    list<int> particles;
    particles.push_back(particle);
    constructNeighborList(s, particles, radius, neighborList);
    int neighb;
    cout << particle << "\n";
    for (auto it = neighborList.begin()->begin(); it != neighborList.begin()->end();it++) {
      neighb = (*it).j;
      cout << neighb << " ";
      for (int d = 0; d < DIM; d++) {FixedDof[neighb*DIM+d] = false;}
    }
    cout << "\n" << flush;
    c.SetFixedDof(FixedDof);

    //now proceed to yield
    CSimpleMinimizer<DIM> miner(c);
    dmat InitTrans;
	  s.GetBox()->GetTransformation(InitTrans);
    int depth = 1;
	  dbl gamma = 0;
    int tempcount = 0;
    dbl dratio = (dbl)ratio;
    int max_steps = 10000;
    for (int i = 0; i < max_steps; i++) {
  		CStaticState<DIM> oldstate = s; //copy
      dbl oldStress = (c.Data.Stress*strain_tensor).trace();
    	dmat strain = (ds / std::pow(dratio,depth - 1.0))*strain_tensor;
      dmat InitTrans;
      c.StdPrepareSystem();
      dbl oldEnergy = c.ComputeMobileEnergy();
      Eigen::VectorXd pos;
      s.GetBox()->GetTransformation(InitTrans);
      InitTrans *= dmat::Identity()+strain;
      s.GetBox()->SetTransformation(InitTrans);
      miner.minimizeFIRE(tol,-1,-1,10000,true);//silent minimization
  	  c.StdPrepareSystem(false);//quiet
    	c.CalculateStdData();
      dbl Energy = c.ComputeMobileEnergy();
      cout << oldEnergy << " " << Energy << " " << c.Data.Energy << "\n";
  	  //if ((c.Data.Stress*strain_tensor).trace() < oldStress) { //go slower
  	  if (Energy < oldEnergy) {
  			if (depth < maxdepth) {
  			  s = oldstate; //go back
          gamma -= ds / std::pow(dratio,depth - 1.0);
          c.StdPrepareSystem(false); //quiet
          c.CalculateStdData();
        	depth += 1;
          tempcount = 0;
  			}
  			else { //we have gotten as close as we care to
          s = oldstate;
          return oldStress;
        }
  		}
      else {
        tempcount += 1;
        if (tempcount >= ratio) {
          tempcount = 0;
          depth = 1;
        }
      }
  	}
    return -1; //should never happen
  }


  dbl localYieldWrite(CStaticState<DIM> & s, CStaticDatabase<DIM> & db, int particle, dbl radius, dmat strain_tensor, dbl ds=1e-4, dbl ratio=5.0, int maxdepth=5, dbl tol=1e-12) {
    //make a computer and pin the particles outside the radius
    CStaticComputer<DIM> c(s);
    int N = s.GetN();
    vector<bool> FixedDof(DIM*N,true);
    for (int d = 0; d < DIM;d++) {FixedDof[particle*DIM+d] = false;}
    list<list<CSimpleNeighbor<DIM> > > neighborList;
    list<int> particles;
    particles.push_back(particle);
    constructNeighborList(s, particles, radius, neighborList);
    int neighb;
    cout << particle << "\n";
    for (auto it = neighborList.begin()->begin(); it != neighborList.begin()->end();it++) {
      neighb = (*it).j;
      cout << neighb << " ";
      for (int d = 0; d < DIM; d++) {FixedDof[neighb*DIM+d] = false;}
    }
    cout << "\n" << flush;
    c.SetFixedDof(FixedDof);

    //now proceed to yield
    CSimpleMinimizer<DIM> miner(c);
    dmat InitTrans;
	  s.GetBox()->GetTransformation(InitTrans);
    int depth = 1;
	  dbl gamma = 0;
    int tempcount = 0;
    dbl dratio = (dbl)ratio;
    int max_steps = 10000;
    for (int i = 0; i < max_steps; i++) {
  		CStaticState<DIM> oldstate = s; //copy
      dbl oldStress = (c.Data.Stress*strain_tensor).trace();
    	dmat strain = (ds / std::pow(dratio,depth - 1.0))*strain_tensor;
      dmat InitTrans;
      c.StdPrepareSystem();
      dbl oldEnergy = c.ComputeMobileEnergy();
      Eigen::VectorXd pos;
      s.GetBox()->GetTransformation(InitTrans);
      InitTrans *= dmat::Identity()+strain;
      s.GetBox()->SetTransformation(InitTrans);
      miner.minimizeFIRE(tol,-1,-1,10000,true);//silent minimization
  	  c.StdPrepareSystem(false);//quiet
    	c.CalculateStdData();
      dbl Energy = c.ComputeMobileEnergy();
      cout << oldEnergy << " " << Energy << " " << c.Data.Energy << "\n";
  	  //if ((c.Data.Stress*strain_tensor).trace() < oldStress) { //go slower
  	  if (Energy < oldEnergy) {
  			if (depth < maxdepth) {
  			  s = oldstate; //go back
          gamma -= ds / std::pow(dratio,depth - 1.0);
          c.StdPrepareSystem(false); //quiet
          c.CalculateStdData();
        	depth += 1;
          tempcount = 0;
  			}
  			else { //we have gotten as close as we care to
          s = oldstate;
          return oldStress;
        }
  		}
      else {
        db.Write(s);
        tempcount += 1;
        if (tempcount >= ratio) {
          tempcount = 0;
          depth = 1;
        }
      }
  	}
    return -1; //should never happen
  }

*/
/*
  dmat shearTensorIterator(int i) {
      int ii = 0;
      dmat strain_tensor = dmat::Zero();
      for(int j = 0; j < DIM; j++) {
        for(int k = j+1; k < DIM; k++) {
          if (ii == i) {
            strain_tensor(j,k) = 0.5;
            strain_tensor(k,j) = 0.5;
          }
          ii++;
        }
      }

      for(int j= 0; j < DIM; j++) {
        if (ii == i) {
          strain_tensor(j,j) = 0.5;
          if (j < DIM-1) {
            strain_tensor(j+1,j+1) = -0.5;
          }
          else { strain_tensor(0,0) = -0.5; }
        }
        ii++;
      }
      return strain_tensor;
  }
*/




}

#endif
