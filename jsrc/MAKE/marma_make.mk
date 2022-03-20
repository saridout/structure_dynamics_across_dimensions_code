

INCLUDE = \
-I$(srcDIR) \
-I/data1/shared/igraham/arpack++/include\
-I/home0/ridout/packages/arpack++/1.2-icc/include \
-I/packages/netcdf/4.3.2/intel-14.0/include \
-I/packages/netcdf/4.4.1_intel-17/include \
-I/usr/global/include/eigen3 \
-I/home/ridout/suitesparse \
-I/packages/boost/1.55.0/include\
-I/projects/ajliu/ridout/packages/arpack++/include\
-I/data1/shared/igraham/netcdf-cxx/4.2/include\
-I/data1/shared/igraham/netcdf-c/4.7.2/include \
-I/data1/shared/SuiteSparse-5.6.0/UMFPACK/Include \
-I/data1/shared/SuiteSparse-5.6.0/include/
#-I/home0/mlav/netcdf/4.3.0-icc/include \
#-I/home0/mlav/netcdf-cxx/4.2-icc/include \
#//packages/netcdf/4.3.2/intel-14.0/
#
#-I//home0/mlav/jamming/arpack++/include \
#-I//home0/mlav/jsrc2/jcode/local/include/eigen3.2.0 \
#-I/data1/jcode/local/include
#-I/usr/global/netcdf-4.1.1-i11/include \
-I/data0/home/cpgoodri/jmodes/cpp/netcdf-4.1.3/include \
-I/data1/jcode/local/include/netcdf4 \

LIBRARY = \
-L/packages/ARPACK \
-L/packages/hdf5/1.8.14/intel-14.0/lib \
-L/packages/netcdf/4.3.2/intel-14.0/lib \
-L/packages/netcdf/cxx4-4.3/lib \
-L/packages/SuiteSparse/4.5.3/lib  \
-L/packages/boost/1.55.0/lib \
-L/projects/eb-racs/p/software/netCDF-C++4/4.3.0-intel-2017a/lib \
-L/data1/shared/igraham/netcdf-cxx/4.2/lib \
-L/data1/shared/igraham/netcdf-c/4.7.2/lib\
-L/usr/lib64/\
-L/usr/lib64/atlas/\
-L/lib64/\
-L/usr/global/lib/
#-L/projects/eb-racs/p/software/SuiteSparse/4.5.5-intel-2017a-METIS-5.1.0/lib \
#-L/home0/mlav/netcdf/4.3.0-icc/lib \
#-L/home0/mlav/netcdf-cxx/4.2-icc/lib \
#-L/home0/mlav/hdf5/1.8.10-patch1-icc/lib \
#-L/data1/jcode/local/lib/static \
-L/data1/shared/igraham/netcdf-c/4.7.2/lib\
#-L/data1/shared/igraham/netcdf-cxx/4.2.1/lib \

#-L//home0/mlav/jsrc2/jamming/lib \
#-L//home0/mlav/jsrc2/jcode/local/lib
#SuiteSparseLINK = -lamd -lcholmod -lcolamd -lccolamd -lcamd -lumfpack -lblas
SuiteSparseLINK = -l:libumfpack.so.5# -lamd -lcolamd
netCDFLINK = -lnetcdf
netCDFLINK = -lnetcdf_c++ -lnetcdf
#hdf5LINK = -lhdf5_hl -lhdf5 -lz
intelLINK = -lm #-lifcore -lm -lifport -lcholmod
arpackLINK =  -l:libgfortran.so.3 -l:libarpack.so.2 -l:libsatlas.so.3 -l:libtatlas.so.3
LINK = $(SuiteSparseLINK) $(netCDFLINK) $(hdf5LINK) $(intelLINK) $(arpackLINK)
