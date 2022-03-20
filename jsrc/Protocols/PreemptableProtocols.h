/*
Easily run on preemtable queue.
write(): write state and any traits needed to tell current state of protocol to file
read(): recover these properties
step(): take single step of this protocol

e.g. suppose protocol is to strain by ever decreasing amounts
0.01, 0.001, 0.00001, 0.0000001....
Then write() needs to not only save current state, but also current strain,
so that we can pick up where we left off if we ever get preempted.
(or run out of walltime)
*/
namespace Protocols {

template <int Dim>
class PreemptableProtocol {

  public: //since I haven't bothered writing get/set functions yet
    std::string statefilename;//place to store backup of state on disk
    std::string auxfilename;//place to store backup of simulation progress
    CStaticState<Dim> * state;
    int stepcount;
    PreemptableProtocol();
    PreemptableProtocol(CStaticState<Dim> * s, std::string sf, std::string af);
    virtual void Read() = 0;
    virtual void Write() = 0;
    virtual int Step() = 0; //return 0 if incomplete, 1 if complete
    void runUntilCompletion() {
      while(!Step()) {Write();}
    }
    void runNSteps(int n) {
      while(stepcount < n) {Step(); Write();}
    }
};

template <int Dim>
PreemptableProtocol<Dim>::PreemptableProtocol() {}

template <int Dim>
PreemptableProtocol<Dim>::PreemptableProtocol(CStaticState<Dim> * s, std::string sf, std::string af) {
  statefilename = sf;
  auxfilename = af;
  state = s;
  //if state and aux files exist, we resume. otherwise do nothing.
  if (file_exists(statefilename) && file_exists(auxfilename)) {
    Read();
  }

}

template <int Dim>
class PreemptableSlowCompressionToPtarg: public PreemptableProtocol<Dim> {
  private:
    dbl Ptarg, dphi, ratio,  tol;
    int maxdepth, depth, tempcount;
    bool nonaffine, resolveRattlers;
  public:
    PreemptableSlowCompressionToPtarg();
    PreemptableSlowCompressionToPtarg(CStaticState<Dim> * s,
      std::string sf, std::string af, dbl _Ptarg=1e-8, dbl _dphi=1e-5,dbl _ratio = 2,
      dbl _tol =1e-12, int _maxdepth = 34, int _depth = 0, int _tempcount = 0): Ptarg(_Ptarg),
      dphi(_dphi),ratio(_ratio), tol(_tol),maxdepth(_maxdepth), depth(_depth) {statefilename = sf;
      auxfilename = af;state = s;if (file_exists(statefilename) && file_exists(auxfilename)) {Read();  }};
    void Read();
    void Write();
    int Step()
};


template <int Dim>
void PreemptableSlowCompressionToPtarg<Dim>::Write(){
  CStaticDatabase<Dim> write(state->GetParticleNumber(),statefilename,NcFile::Replace);
  write.Write(*state);
  std::ofstream stream(auxfilename);
  stream << Ptarg << "\n" << dphi<<"\n"<< ratio<< "\n"<<  tol << "\n";
  stream << maxdepth <<"\n" <<  depth << "\n"<< tempcount;
  stream << nonaffine << "\n" << resolveRattlers << "\n";
  stream.flush();
}

template <int Dim>
void PreemptableSlowCompressionToPtarg<Dim>::Read(){
  CStaticDatabase<Dim> read(state->GetParticleNumber(),statefilename,NcFile::ReadOnly);
  read.Read(*state,0);
  std::ifstream stream(auxfilename);
  stream >> Ptarg;
  stream >> dphi;
  stream >> ratio;
  stream >> tol;
  stream >> maxdepth;
  stream >> depth;
  stream >> tempcount;
  //temporarily allow for "old" data
  stream >> nonaffine;
  stream >> resolveRattlers;
}


template <int Dim>
int PreemptableSlowCompressionToPtarg<Dim>::Step(){
  CStaticState<DIM> oldstate = *state; //copy
  dbl phi = state->GetPackingFraction();
  CStaticComputer<DIM> c(*state);
  CSimpleMinimizer<DIM> miner(c);
  if (nonaffine) {c.StdPrepareSystem(); c.CalculateStdData();}
  if (nonaffine && c.Data.cijkl.CalculateBulkModulus() > 0.0001) {
    dmat strain = dPhiToStrain(dphi/std::pow((dbl)ratio,depth-1.0),phi,DIM)*dmat::Identity();
    c.ShearWithNonAffine(strain);
  }
  else
    state->SetPackingFraction(phi + dphi/std::pow((dbl)ratio,depth-1.0));
  if (resolveRattlers) {
    c.ResolveRattlers();
  }
  miner.minimizeFIRE(tol);
  int jammed = 0;
  if(! c.StdPrepareSystem()) {
    c.CalculateStdData();
    jammed = c.Data.cijkl.CalculateBulkModulus() > 0.0001;
  }
  c.CalculateStdData();
  stepcount += 1;
  if (c.Data.Pressure > Ptarg && jammed) { //go slower
    if (depth < maxdepth) {
      *state = oldstate; //go back
      depth += 1;
      tempcount = 0;
      return 0;
    }
    else { //we have gotten as close as we care to
      *state = oldstate;
      return 1;
    }
  }
  else {
    tempcount += 1;
    if (tempcount >= ratio) {
      tempcount = 0;
      depth = 1;
    }
    return 0;
  }
}

template <int Dim>
class PreemptableEnthalpySafeShear: public PreemptableProtocol<Dim> {
  private:
    dbl P, dgamma, ratio,  tol;
    int maxdepth, depth, tempcount;
    bool nonaffine, resolveRattlers;
  public:
    PreemptableEnthalpySafeShear();
    PreemptableEnthalpySafeShear(CStaticState<Dim> * s,
      std::string sf, std::string af, dbl _P=1e-2, bool _na=true, bool _rr = false, dbl _dgamma=1e-5,dbl _ratio = 2,
      dbl _tol =1e-12, int _maxdepth = 34, int _depth = 0, int _tempcount = 0): P(_P),
      nonaffine(_na),resolveRattlers(_rr), dgamma(_dgamma),ratio(_ratio), tol(_tol),
      maxdepth(_maxdepth), depth(_depth) {statefilename = sf; auxfilename = af;
      state = s;if (file_exists(statefilename) && file_exists(auxfilename)) {Read();  }};
    void Read();
    void Write();
    int Step()
};

template <int Dim>
void PreemptableEnthalpySafeShear<Dim>::Write(){
  CStaticDatabase<Dim> write(state->GetParticleNumber(),statefilename,NcFile::Replace);
  write.Write(*state);
  std::ofstream stream(auxfilename);
  stream << stepcount << "\n";
  stream << P << "\n" << dgamma <<"\n"<< ratio<< "\n"<<  tol << "\n";
  stream << maxdepth <<"\n" <<  depth << "\n"<< tempcount << "\n";
  stream << nonaffine << "\n" << resolveRattlers << "\n";
  stream.flush();
}

template <int Dim>
void PreemptableEnthalpySafeShear<Dim>::Read(){
  CStaticDatabase<Dim> read(state->GetParticleNumber(),statefilename,NcFile::ReadOnly);
  read.Read(*state,0);
  std::ifstream stream(auxfilename);
  stream >> stepcount;
  stream >> P;
  stream >> dgamma;
  stream >> ratio;
  stream >> tol;
  stream >> maxdepth;
  stream >> depth;
  stream >> tempcount;
  stream >> nonaffine;
  stream >> resolveRattlers;
}

template <int Dim>
int PreemptableEnthalpySafeShear<Dim>::Step(){
  CStaticState<DIM> oldstate = *state; //copy
  dmat InitTrans;
  state->GetBox()->GetTransformation(InitTrans);
  dmat strain_tensor = dmat::Zero();
  strain_tensor(1,0) = 1.0;
  CBoxyComputer<DIM> c(*state,P);
  CSimpleMinimizer<DIM> miner(c);
  dbl oldStress = c.Data.Stress(0,1);
  dmat strain = (dgamma / std::pow(5.0,depth - 1.0))*strain_tensor;
  if (nonaffine ) {
    c.StdPrepareSystem();
    c.ShearWithNonAffine(strain);
  }
  else
    abort();//don't feel like handling this case right now
  if (resolveRattlers) {
    cout << "resolving rattlers"<<"\n";
    c.ResolveRattlers();
  }
  miner.minimizeFIRE(tol);
  int jammed = 0;
  if(! c.StdPrepareSystem()) {
    c.CalculateStdData();
    jammed = c.Data.cijkl.CalculateBulkModulus() > 0.0001;
  }
  c.CalculateStdData();
  stepcount += 1;
  if (c.Data.Stress(0,1) < (1-0.01*sgn(oldStress))*oldStress) { //go slower
    if (depth < maxdepth) {
      *state = oldstate; //go back
      depth += 1;
      tempcount = 0;
      return 0;
    }
    else { //we have gotten as close as we care to
      *state = oldstate;
      return 1;
    }
  }
  else {
    tempcount += 1;
    if (tempcount >= ratio) {
      tempcount = 0;
      depth = 1;
    }
    return 0;
  }
}
}
