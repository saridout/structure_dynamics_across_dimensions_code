#ifndef STD_DATA_H
#define STD_DATA_H

#include "../Computers/cijkl.h"
#include "../Computers/MatrixInterface.h"

namespace LiuJamming
{

template<int Dim>
class CStdData
{
	typedef Eigen::Matrix<dbl,Dim,Dim> dmat;
	typedef Eigen::Matrix<dbl,Dim,1> dvec;
public:
	int NPp;		//!<Number of particles participating (i.e. not rattlers)
	int Nc;			//!<Number of contacts
	dbl Volume;

	dbl Energy;
	dbl Pressure;
	dmat Stress;	//!<The stress tensor
	dmat Fabric;	//!<The fabric tensor
	dbl MaxGrad;
  #if DIM < 4
	cCIJKL<Dim> cijkl;
  #endif
	MatrixInterface<dbl> H;
	MatrixInterface<dbl> GH; //SR original
	
	inline int Nciso()		const { return Dim*(NPp-1); };
	inline int NcmNciso()	const { return Nc - Nciso(); };
	inline dbl Z()			const { return 2.*((dbl)Nc)/((dbl)NPp); };
	inline dbl deltaZ()		const { return 2.*((dbl)NcmNciso())/((dbl)NPp); };
	
	//SR
	inline int Ncisoprime() const { return Nciso() + 1 ;};
	inline int NcmNcisoprime() const {return Nc - Ncisoprime(); };
	inline dbl deltaZprime() const { return 2.*((dbl)NcmNcisoprime()) / ((dbl)NPp);};
  inline dbl devStressSquared() const { return ((Stress - Pressure*Eigen::MatrixXd::Identity(Dim,Dim))*(Stress - Pressure*Eigen::MatrixXd::Identity(Dim,Dim))).trace();}	

	CStdData()
	{
		SetZero();
	};

	void SetZero()
	{
		NPp = Nc = 0;
		Volume = Energy = Pressure = MaxGrad = 0.;
		Stress = dmat::Zero();
		Fabric = dmat::Zero();
		#if DIM < 4
    cijkl.SetZero();
    #endif
		//Set the matrix to zero???
	};

	void Print()
	{
		printf("----------------------------------------------------------------\n");
		printf("Printing standard data:\n");
		printf("\tNPp         = %i\n", NPp);
		printf("\tNc          = %i\n", Nc);
		printf("\tZ           = %f\n", Z());
		printf("\tNcmNciso    = %i\n", NcmNciso());
		printf("\tDelta Z     = % e\n", deltaZ());
		printf("\tEnergy      = % e\n", Energy);
		printf("\tPressure    = % e\n", Pressure);
		printf("\tMaxGrad     = % e\n", MaxGrad);
		printf("\tStress Tensor: \n");
		for(int d1=0; d1<Dim; ++d1)
		{
			printf("\t             ");
			for(int d2=0; d2<Dim; ++d2) printf(" % e  ", Stress(d1,d2));
			printf("\n");
		}
		printf("\tFabric Tensor: \n");
		for(int d1=0; d1<Dim; ++d1)
		{
			printf("\t             ");
			for(int d2=0; d2<Dim; ++d2) printf(" % e  ", Fabric(d1,d2));
			printf("\n");
		}
		printf("\tElastic Constants: \n");
    #if DIM < 4
		Eigen::MatrixXd Cik = cijkl.get_Cik();
    
		char dot[32];
		sprintf(dot,"      .      ");
		for(int d1=0; d1<Cik.rows(); ++d1)
		{
			printf("\t             ");
			for(int d2=0; d2<d1; ++d2) printf(" %s  ", dot);
			for(int d2=d1; d2<Cik.cols(); ++d2) printf(" % e  ", Cik(d1,d2));
			printf("\n");
		}
		printf("\tG = % e\n", cijkl.CalculateShearModulus());
		printf("\tB = % e\n", cijkl.CalculateBulkModulus());
	  #endif
  	printf("----------------------------------------------------------------\n\n");
	}
	void PrintLine(FILE *outfile)
	{
		//Data: NPp, Nc, NcmNciso, Z, DeltaZ, Volume, Energy, Pressure, MaxGrad, G, B
		//        1   2         3  4       5       6       7         8        9 10 11
  #if DIM < 4	
	fprintf(outfile, "%i, %i, %i, %e, %e, %e, %e, %e, %e, %e, %e,\n", NPp, Nc, NcmNciso(), Z(), deltaZ(), Volume, Energy, Pressure, MaxGrad, cijkl.CalculateShearModulus(), cijkl.CalculateBulkModulus());
	#else
  	fprintf(outfile, "%i, %i, %i, %e, %e, %e, %e, %e, %e, %e, %e,\n", NPp, Nc, NcmNciso(), Z(), deltaZ(), Volume, Energy, Pressure, MaxGrad);
	
  #endif
  }
  #if DIM < 4
	void wavefreqs(dvec &omega,dvec const &k) //Computes continuum wave frequencies at wavevector k according to Cijkl and Sij
	{
		dmat mat; 
		cijkl.CalculateCijklVjVk(mat,k);
		dbl temp = k.dot(Stress*k);
		mat += temp*dmat::Identity();

		
		Eigen::JacobiSVD<dmat> svd(mat); //I may be wrong, but my understanding is that this is more accurate than, say EigenSolver
		
		omega = ((((svd.singularValues()).array()).sqrt()).matrix());

		dbl density = (dbl) NPp / Volume;
		omega = omega / sqrt(density);
		
	}

  #endif //DIM < 4
};

}

#endif //STD_DATA_H

