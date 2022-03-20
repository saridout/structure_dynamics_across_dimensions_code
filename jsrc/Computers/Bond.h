#ifndef BOND_H
#define BOND_H

#include "../Resources/std_include.h"

namespace LiuJamming
{

//! Class to store a single bond.

template<int Dim>
class CBond
{
	typedef Eigen::Matrix<dbl, Dim, 1> dvec;
	typedef Eigen::Matrix<dbl, Dim, Dim> dmat;
public:
	int	i;		//!<Index i.
	int j;		//!<Index j.
	dvec r;		//!<A dvec pointing from i to j.
	dbl rlen;	//!<Distance between i and j.
	dbl E;		//!<Potential energy, V, stored in the bond.
	dbl g;		//!<First derivative of V w.r.t. rlen.
	dbl k;		//!<Second derivative of V w.r.t rlen.


//! @name Constructors and Operators
///@{
	CBond(int _i=0, int _j=0);													//!<Default constructor
	CBond(int _i, int _j, dbl _rlen, dbl _E, dbl _g, dbl _k, const dvec &_r);	//!<Constructor with full data
	CBond(const CBond<Dim> &src);												//!<Copy constructor
	CBond<Dim>& operator=(const CBond<Dim> &src);								//!<Copy operator

///@}

//! @name Misc.
///@{
	void CalculateMatrixBlocks(dmat &Fij, dmat &Kij) const;						//!<Calculate blocks for the hessian.

	dbl CalculateEnergyChange(dvec const &DeltaR, dbl unstress_coeff=1., dbl stress_coeff=1.) const;

	void print() const;															//!<Print the bond to the terminal.

///@}
};

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   IMPLEMENTATION   ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<int Dim>
CBond<Dim>::CBond(int _i, int _j)
	:i(_i), j(_j), rlen(0.), E(0.), g(0.), k(0.)
{
	r = dvec::Zero(); 
};

template<int Dim>
CBond<Dim>::CBond(int _i, int _j, dbl _rlen, dbl _E, dbl _g, dbl _k, const dvec &_r)
	:i(_i), j(_j), rlen(_rlen), E(_E), g(_g), k(_k)
{
	r = _r; 
};

template<int Dim>
CBond<Dim>::CBond(const CBond<Dim> &src) 
{
	(*this) = src; 
};

template<int Dim>
CBond<Dim> &CBond<Dim>::operator=(const CBond<Dim> &src)
{
	if(this != &src)
	{
		i = src.i;
		j = src.j;
		r = src.r;
		rlen = src.rlen;
		E = src.E;
		g = src.g;
		k = src.k;
	}
	return *this;
}

/**
 *	@param[out] Fij The stressed component of the matrix block.
 *	@param[out] Kij The unstressed component of the matrix block.
 */
template<int Dim>
void CBond<Dim>::CalculateMatrixBlocks(dmat &Fij, dmat &Kij) const
{
	Fij = -g*(dmat::Identity() - r*(r.transpose())/POW2(rlen))/rlen;
	Kij = -k*r*(r.transpose())/POW2(rlen);
}

template<int Dim>	
dbl CBond<Dim>::CalculateEnergyChange(dvec const &DeltaR, dbl unstress_coeff, dbl stress_coeff) const
{
	dbl gt, kt, DeltaRparallel2, DeltaRperp2;
	
	gt = g*stress_coeff;
	kt = k*unstress_coeff;
	DeltaRparallel2 = POW2(DeltaR.dot(r)/rlen);
	DeltaRperp2     = DeltaR.squaredNorm() - DeltaRparallel2;

	return 0.5*(kt*DeltaRparallel2 + gt*DeltaRperp2/rlen);
}


template<int Dim>
void CBond<Dim>::print() const
{
	printf("bond between %5i and %5i: r = ", i, j);
	for(int dd=0; dd<Dim; ++dd)
		printf("% e ", r[dd]);
	printf("rlen =% e, E =% e, g =% e, k =% e\n", rlen, E, g, k);
}











}

#endif //BOND_H

