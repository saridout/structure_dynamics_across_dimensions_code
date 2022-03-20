Simulation code used in "Correlation of plastic events with local structure in jammed packings across spatial dimensions".

To compile, first create a file analogous to ``marma_make.mk`` for your own computer / cluster, specifying the locations of required external code (Suitesparse, ARPACK, NetCDF). Edit ``dirs.h`` to indicate the directories where you want to save states / small datafiles.

Then edit ``mains/makefile`` to select which of the files in ``mains`` you wish to compile, and then run e.g. ``dim=2 make`` to compile the 2d version of the script.
