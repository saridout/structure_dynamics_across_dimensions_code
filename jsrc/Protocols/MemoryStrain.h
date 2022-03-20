/*
 * A series of classes to implement an "elastic memory" based strain protocol.
 * Use previous nonaffine response to guess future nonaffine response before minimizing.
 * Preferred over computer NonAffineMove for large systems due to poor scaling of matrix eq. solving.
 * Sean Ridout, March 2019
 */
#include "State/StaticState.h"
#include "Computers/StaticComputer.h"
#include "Minimization/minimizer.h"
#include "Resources/defs.h"
#include "Resources/Helper.h"
using namespace LiuJamming;
namespace Protocols {

template <int Dim>
class SingleMemoryStrain {
public:
  CStaticState<Dim> * state;
  CStaticComputer<Dim> * comp;
  CSimpleMinimizer<Dim> * miner;
  Eigen::VectorXd grad;
  Eigen::Matrix<dbl,Dim,Dim> strain_mat;  
  SingleMemoryStrain(CStaticState<Dim> * state_, CStaticComputer<Dim> * comp_, CSimpleMinimizer<Dim> * miner_, Eigen::Matrix<dbl,Dim,Dim> strain_mat_);
  void strain_step(dbl strain_mag );
  void forget();
};

template <int Dim>
class DoubleMemoryStrain {
public:
  CStaticState<Dim> * state;
  CSimpleMinimizer<Dim> * miner;
  Eigen::VectorXd grad;
  Eigen::VectorXd gradgrad;
  Eigen::Matrix<dbl,Dim,Dim> strain_mat;
  DoubleMemoryStrain(CStaticState<Dim> * state_, CSimpleMinimizer<Dim> * miner_, Eigen::Matrix<dbl,Dim,Dim> strain_mat_);
  void strain_step(dbl strain_mag );
  void forget();
};

template<int Dim>
SingleMemoryStrain<Dim>::SingleMemoryStrain(CStaticState<Dim> * state_, CStaticComputer<Dim> * comp_, CSimpleMinimizer<Dim> *miner_,Eigen::Matrix<dbl,Dim,Dim> strain_mat_) {
  state = state_;
  comp = comp_;
  miner = miner_;
  strain_mat = strain_mat_;
  grad = Eigen::VectorXd::Zero(state->GetN()*DIM);
}

template <int Dim>
void SingleMemoryStrain<Dim>::strain_step(dbl strain_mag) {
  Eigen::VectorXd old_pos;
  state->GetPositionsVirtual(old_pos);

  Eigen::Matrix<dbl,Dim,Dim> box;
  state->GetBox()->GetTransformation(box);
  box *= dmat::Identity()+strain_mag*strain_mat;
  state->GetBox()->SetTransformation(box);
  state->MoveParticlesVirtual(grad*strain_mag);
  //comp->ComputeBondList(comp->Bonds);
  //comp->ResolveRattlers();
  miner->minimizeFIRE();

  //now update grad
  state->GetPositionsVirtual(grad);
  grad -=old_pos;
  for (int i = 0; i < grad.innerSize(); i++) { 
      if (grad(i) > 0.5) { grad(i) -= 1.0;}
      if (grad(i) < -0.5) {grad(i) += 1.0;}
  }
  grad /= strain_mag;
 
}

template <int Dim>
DoubleMemoryStrain<Dim>::DoubleMemoryStrain(CStaticState<Dim> * state_, CSimpleMinimizer<Dim> * miner_,Eigen::Matrix<dbl,Dim,Dim> strain_mat_) {
  state = state_;
  miner = miner_;
  strain_mat = strain_mat_;
  grad = Eigen::VectorXd::Zero(state->GetN()*Dim);
  gradgrad = grad;
}

template <int Dim>
void DoubleMemoryStrain<Dim>::strain_step(dbl strain_mag) {
  Eigen::VectorXd old_pos;
  state->GetPositionsVirtual(old_pos);

  Eigen::Matrix<dbl,Dim,Dim> box;
  state->GetBox()->GetTransformation(box);
  box *= dmat::Identity()+strain_mag*strain_mat;
  state->GetBox()->SetTransformation(box); 
  state->MoveParticlesVirtual((grad+strain_mag*gradgrad)*strain_mag);
  miner->minimizeFIRE();

  //now update grad
  gradgrad = -grad;
  state->GetPositionsVirtual(grad);
  grad -=old_pos;
  for (int i = 0; i < grad.innerSize(); i++) { 
      if (grad(i) > 0.5) { grad(i) -= 1.0;}
      if (grad(i) < -0.5) {grad(i) += 1.0;}
  }
  grad /= strain_mag;
  //now update gradgrad
  gradgrad += grad;
  gradgrad /= strain_mag;
 
}

template <int Dim>
void DoubleMemoryStrain<Dim>::forget() {
  grad *= 0;
  gradgrad *= 0;
}

template <int Dim>
void SingleMemoryStrain<Dim>::forget() {
  grad *= 0;
}

} //end namespace

