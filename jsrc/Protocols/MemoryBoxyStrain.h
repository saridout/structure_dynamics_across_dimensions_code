/*
 * A series of classes to implement an "elastic memory" based strain protocol.
 * Use previous nonaffine response to guess future nonaffine response before minimizing.
 * Preferred over computer NonAffineMove for large systems due to poor scaling of matrix eq. solving.
 * Sean Ridout, March 2019
 */
#include "State/StaticState.h"
#include "Minimization/minimizer.h"
#include "Resources/defs.h"
#include "Resources/Helper.h"
using namespace LiuJamming;
namespace Protocols {

template <int Dim>
class DoubleMemoryBoxyStrain {

private:
  CStaticState<Dim> * state;
  CSimpleMinimizer<Dim> * miner;
  Eigen::VectorXd grad;
  Eigen::VectorXd gradgrad;
  Eigen::Matrix<dbl,Dim,Dim> strain_mat;
  long N;
  double volgrad;
  double volgradgrad;
public:
  DoubleMemoryBoxyStrain(CStaticState<Dim> * state_, CSimpleMinimizer<Dim> * miner_, Eigen::Matrix<dbl,Dim,Dim> strain_mat_);
  void strain_step(dbl strain_mag );
  void forget();
};

template <int Dim>
DoubleMemoryBoxyStrain<Dim>::DoubleMemoryBoxyStrain(CStaticState<Dim> * state_, CSimpleMinimizer<Dim> * miner_,Eigen::Matrix<dbl,Dim,Dim> strain_mat_) {
  state = state_;
  miner = miner_;
  strain_mat = strain_mat_;
  N = state->GetN();
  grad = Eigen::VectorXd::Zero(N*Dim);
  gradgrad = grad;
  double volgrad = 0.0;
  double volgradgrad = 0.0;
}

template <int Dim>
void DoubleMemoryBoxyStrain<Dim>::strain_step(dbl strain_mag) {
  Eigen::VectorXd old_pos;
  
  state->GetPositionsVirtual(old_pos);
  double old_vol = state->GetVolume();

  Eigen::Matrix<dbl,Dim,Dim> box;
  state->GetBox()->GetTransformation(box);
  box *= dmat::Identity()+strain_mag*strain_mat;
  state->GetBox()->SetTransformation(box); 
  
  state->MoveParticlesVirtual((grad+strain_mag*gradgrad)*strain_mag);
  state->SetVolume(old_vol+(volgrad+strain_mag*volgradgrad)*strain_mag);
  miner->minimizeFIRE();

  //now update grad
  gradgrad = -grad;
  volgradgrad = -volgrad;
  state->GetPositionsVirtual(grad);
  state->GetBox()->MoveParticles(grad,-old_pos); //this is virtual by default
  volgrad = state->GetVolume() - old_vol;
  grad /= strain_mag;
  volgrad /= strain_mag;
  //now update gradgrad
  state->GetBox()->MoveParticles(gradgrad,grad);
  volgradgrad += volgrad;
  gradgrad /= strain_mag;
  volgradgrad /= strain_mag;
}

template <int Dim>
void DoubleMemoryBoxyStrain<Dim>::forget() {
  grad *= 0;
  gradgrad *= 0;
  volgrad = 0.0;
  volgradgrad = 0.0;

}
/*
template <int Dim>
class TripleMemoryBoxyStrain {

private:
  CStaticState<Dim> * state;
  CSimpleMinimizer<Dim> * miner;
  Eigen::VectorXd grad;
  Eigen::VectorXd gradgrad;
  Eigen::VectorXd grad3;
  Eigen::Matrix<dbl,Dim,Dim> strain_mat;
  long N;
  double volgrad;
  double volgradgrad;
  double vgrad3
public:
  TripleMemoryBoxyStrain(CStaticState<Dim> * state_, CSimpleMinimizer<Dim> * miner_, Eigen::Matrix<dbl,Dim,Dim> strain_mat_);
  void strain_step(dbl strain_mag );
  void forget();
};

template <int Dim>
TripleMemoryBoxyStrain<Dim>::TripleMemoryBoxyStrain(CStaticState<Dim> * state_, CSimpleMinimizer<Dim> * miner_,Eigen::Matrix<dbl,Dim,Dim> strain_mat_) {
  state = state_;
  miner = miner_;
  strain_mat = strain_mat_;
  N = state->GetN();
  grad = Eigen::VectorXd::Zero(N*Dim);
  gradgrad = grad;
  grad3 = grad
  volgrad = 0.0;
  volgradgrad = 0.0;
  vgrad3 = 0.0;
}

template <int Dim>
void TripleMemoryBoxyStrain<Dim>::strain_step(dbl strain_mag) {
  Eigen::VectorXd old_pos;
  
  state->GetPositionsVirtual(old_pos);
  double old_vol = state->GetVolume();

  Eigen::Matrix<dbl,Dim,Dim> box;
  state->GetBox()->GetTransformation(box);
  box *= dmat::Identity()+strain_mag*strain_mat;
  state->GetBox()->SetTransformation(box); 
  
  state->MoveParticlesVirtual((grad+strain_mag*gradgrad)*strain_mag);
  state->SetVolume(old_vol+(volgrad+strain_mag*volgradgrad)*strain_mag);
  miner->minimizeFIRE();

  //now update grad
  gradgrad = -grad;
  state->GetPositionsVirtual(grad);
  state->GetBox()->MoveParticles(grad,-old_pos); //this is virtual by default
  volgrad = state->GetVolume() - old_vol;
  grad /= strain_mag;
  volgrad /= strain_mag;
  //now update gradgrad
  state->GetBox()->MoveParticles(gradgrad,grad);
  volgradgrad += volgrad;
  gradgrad /= strain_mag;
  volgradgrad /= strain_mag;
}

template <int Dim>
void TripleMemoryBoxyStrain<Dim>::forget() {
  grad *= 0;
  gradgrad *= 0;

}
*/
} //end namespace

