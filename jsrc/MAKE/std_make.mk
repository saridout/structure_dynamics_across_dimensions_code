srcDIR = $(DIR)

srcOBJGQS = \
Computers/cijkl.o \
Computers/UmfpackInterface.o \
Resources/Resources.o \
Resources/Exception.o \
Resources/index_map.o \
Resources/RNG_taus.o 

#Compiler
CPP=g++
CC=gcc
FF=gfort


#THE VARIABLE "dim" MUST BE DEFINED IN THE MAIN MAKEFILE

#Compiler flags
#I_ACCURACY_FLAGS = -fp-model precise -prec-div -prec-sqrt
ifeq ($(COMPUTER_NAME),$(Marma_NAME))
I_OPTIMIZATION_FLAGS =  -O3 -fPIC #-xhost -ipo
else
I_OPTIMIZATION_FLAGS = -03 -xhost -ipo
endif

ifdef DEBUG
I_OPTIMIZATION_FLAGS = -g -O1
endif

FLAGS = $(I_OPTIMIZATION_FLAGS) $(I_ACCURACY_FLAGS) #-opt-report-file opt_report.txt
FFLAGS = $(FLAGS)
CFLAGS = $(FLAGS)  -fpermissive -std=c++17 -D DIM=$(dim) -D $(COMPUTER_NAME) #-wr21 -wr279 -wr1125 #-wr418  #-wr1125 -wr2196 -wr2536
LinkFLAGS = $(FLAGS)


ifeq ($(COMPUTER_NAME),$(Walnut_NAME))
include $(DIR)/MAKE/walnut_make.mk
else
ifeq ($(COMPUTER_NAME),$(Fiji_NAME))
include $(DIR)/MAKE/fiji_make.mk
else
ifeq ($(COMPUTER_NAME),$(Oregon_NAME))
include $(DIR)/MAKE/oregon_make.mk
else
ifeq ($(COMPUTER_NAME),$(Talapas_NAME))
include $(DIR)/MAKE/talapas_make.mk
endif
ifeq ($(COMPUTER_NAME),$(Marma_NAME))
include $(DIR)/MAKE/marma_make.mk
endif
endif
endif
endif



FRULE = $(FF)  $(FFLAGS) $(INCLUDE) -c -o $@ $<
CRULE = $(CPP) $(CFLAGS) $(INCLUDE) -c -o $@ $<
ORULE = $(CPP) $(CFLAGS) -o $@ $(OBJGQS) $(LIBRARY) $(LINK)


StandardDependencies = \
	$(srcDIR)/Boundaries/*.h \
	$(srcDIR)/Computers/*.h \
	$(srcDIR)/Database/*.h \
	$(srcDIR)/Minimization/*.h \
	$(srcDIR)/Potentials/*.h \
	$(srcDIR)/Resources/*.h \
	$(srcDIR)/State/*.h



#    $@  means "the target"
#    $<  means "whatever the dependencies are"



