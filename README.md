Code for the paper "Correlation of plastic events with local structure in jammed packings across spatial dimensions".

The same code has also been used previously in APL Materials 9, 021107 (2021); https://doi.org/10.1063/5.0035395

#jsrc
This is the simulation code, based on https://github.com/cpgoodri/jsrc, with small modifications to allow for simulations in d>3, code to minimize enthalpy instead of energy in order to fix the pressure, and various algorithms for more efficient quasistatic shear while carefully approaching instabilities with a finer strain step.

#lib_persistent_homology
Persistent homology code of Jason Rocks, currently not published in generality but first used in DOI: 10.1103/PhysRevResearch.2.033234

#lib_softness
Wrapper of the persistent homology code to specialize in describing the structure of Delaunay triangulations, and read in the states produced by the simulation code. Also contains scripts to do specific analyses.
