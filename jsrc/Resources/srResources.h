#ifndef SRRESOURCES

#define SRRESOURCES

#include <sys/time.h>
#include <ctime>
typedef long long int64; typedef unsigned long long uint64;

//determine s such that strain tensor s*Identity --> desired change in phi
dbl dPhiToStrain(dbl dphi, dbl phi,int Dim) {
  assert(dphi < phi);
  return std::pow(1+dphi/phi,1.0/Dim) - 1.0;
}


uint64 GetTimeMs64()
{
/* Linux */
 struct timeval tv;

 gettimeofday(&tv, NULL);

 uint64 ret = tv.tv_usec;
 /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
 ret /= 1000;

 /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
 ret += (tv.tv_sec * 1000);

 return ret;
}


string d_to_string(dbl d)
{
	ostringstream ss;
	ss << d;
	return ss.str();
}

string i_to_string(int i) {
  ostringstream ss;
  ss << i;
  return ss.str();

}


//a consistent sign choice for a vector orthogonal to a vector in 2D
Eigen::Vector2d orthvec(Eigen::Vector2d n)
{
	Eigen::Vector2d output;
	output(0) = -n(1);
	output(1) = n(0);

	output = output / output.norm();

	return output;
}

dbl mean(vector<dbl> const &vec)
{
	dbl output = 0;

	for(int i =0; i < vec.size(); i++)
	{
		output += vec[i];
	}
	output = output / (dbl) vec.size();
	return output;
}
template<typename T>
T mean(vector<T> const &vec)
{
	T output = 0.0*vec[0];
	for(int i = 0; i < vec.size(); i++)
	{
		output += vec[i];
	}
	output = output / (dbl)vec.size();
	return output;
}

dbl sdev(vector<dbl> const &vec, dbl avg)
{
	dbl output = 0;
	for(int i = 0; i < vec.size(); i++)
	{
		output += POW2(vec[i] - avg);
	}
	output = sqrt(output / (dbl) vec.size());

	return output;
}

dbl sdev(vector<dbl> const &vec)
{
	dbl avg = mean(vec);
	return sdev(vec, avg);
}
inline bool isInteger(const std::string & s)
{
   if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false ;

   char * p ;
   strtol(s.c_str(), &p, 10) ;

   return (*p == 0) ;
}

Eigen::Vector2i read2vec(string a)
{
	stringstream ss(a);
	string buf;
	ss >> buf;
	assert(isInteger(buf));
	Eigen::Vector2i out;
	out(0) = atoi(buf.c_str());
	ss >> buf;
	out(1) = atoi(buf.c_str());
	return out;
}

void read2vecmat(string a, Eigen::Vector2i &v, Eigen::Matrix2d &mat)
{
	stringstream ss(a);
	string buf;
	ss >> buf;
	assert(isInteger(buf));
	v(0) = atoi(buf.c_str());
	ss >> buf;
	v(1) = atoi(buf.c_str());
	ss>> buf;
	mat(0,0) = atof(buf.c_str());
	ss >> buf;
	mat(0,1) = atof(buf.c_str());
	ss >> buf;
	mat(1,0) = atof(buf.c_str());
	ss >> buf;
	mat(1,1) = atof(buf.c_str());
}

void pinv(Eigen::MatrixXd const & A, Eigen::MatrixXd &pinvA,dbl dim)
{
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
	Eigen::VectorXd invD = solver.eigenvalues();
	for(int i =0; i < invD.innerSize(); i++)
	{
		if((abs(invD(i)) < 1e-10)||(i < dim))
			invD(i) =0; //remove zero modes
		else
			invD(i) = 1.0/invD(i); //invert the rest
	}
	Eigen::MatrixXd invDmat = invD.asDiagonal();
	Eigen::MatrixXd V = solver.eigenvectors();

	//the punchline
	pinvA = V*invDmat*V.inverse();
	//NOTE: I don't know if this will compute the inverse of V in an efficient way; it might be worth investigating
}

//yes, this isn't written in a very safe/nice way -it secretly only works if T is an Eigen::Vectortype<dbl>
template<typename T>
string vectorstring(T const &input)
{
	string output = "[ ";
	int N = input.innerSize();
	for(int i =0; i < N-1; i++)
		output += d_to_string(input(i)) + ", ";
	output += d_to_string(input(N-1)) + "]";
	return output;
}

inline bool file_exists (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

//for sorting doubles in descending order and keeping track of indices, for example... or sorting objects based on a single value
template<typename T>
bool greatercomp(std::pair<T,dbl>const &l,std::pair<T,dbl> const &r)
{
	return l.second > r.second;
}
template<typename T>
void sortdescending(vector <std::pair<T,dbl> > &v)
{
	std::sort(v.begin(),v.end(),greatercomp<T>);
}

//same but ascending
template<typename T>
bool lessercomp(std::pair<T,dbl> const &l, std::pair<T,dbl> const &r)
{
	return l.second < r.second;
}
template<typename T>
void sortascending(vector <std::pair<T,dbl> > &v)
{
	std::sort(v.begin(),v.end(),lessercomp<T>);
}

bool isdir(string const &a)
{
	struct stat sb;
	return stat(a.c_str(),&sb)==0 && S_ISDIR(sb.st_mode);
}

#ifndef FIJI //fucking out of date icpc won't let me use C++11
dbl interpolate(const vector<dbl> &X, const vector<dbl> &Y, dbl a) {
  assert(X.size() == Y.size());
  //X IS ASSUMED TO BE SORTED
  auto it = std::find_if(X.begin(),X.end(), [a](dbl b) {
            return (b > a);
        });
  //we will have problems if it is beginning or end
  assert(it != X.begin());
  assert(it != X.end());

  int i = it - X.begin();
  //now interpolate
  return  Y[i-1] + (a-X[i-1])*(Y[i] - Y[i-1])/ (X[i] - X[i-1]);

}
#endif

/*
template<int Dim>
std::string get_harmonic_state_filename<Dim>(string type, string N_string,string Lp_string) {

  string output;
  if (type=="cdb") {
    if( Dim == 2) {
      std::string dir = (std::string)CPG_STATE_DIR + "2d_harmonic_poly_uniform/";
      output = dir + "statedb_N0"+N_string+"_Lp-" + Lp_string + "000.nc";
    }
    else {
      std::string dir = (std::string)CPG_STATE_DIR+"3d_harmonic_bi/";
      output = dir + "statedb_N0"+N_string+"_Lp-"+Lp_string+"000.nc";
    }
  }

  if (type=="cdb2") {
    std::string dir = CPG_STATE_DIR2;
  	if (Dim == 2) {
  		dir += "2d_harmonic/poly_uniform/";
  	}
  	else {
  		dir += "3d_harmonic/bi/";
  	}
  	output = dir + "state_N0000"+N_string+"_Lp-"+Lp_string+"000.nc";

  }

  if (type=="dh") {
    std::string dir = DH_STATE_DIR;
    if (Dim ==2)
  }

  else {
    throw Exception("Input \"type\" does not correspond to a supported state directory type.");
  }


  return output

}
*/

#endif
