#DIR = /data0/home/cpgoodri/jcode/ssdirectional/jsrc
DIR = /home1/ridout/softdimcode/jsrc
DEBUG=

name=Hpmemsafeshear
obj=$(name).o

#define some computer names
Walnut_NAME=walnut
Fiji_NAME=fiji
Talapas_NAME=talapas
Marma_NAME=marmalade
#set this to the computer name you are using
COMPUTER_NAME=marmalade


prjDIR = .
prjOBJGQS = \
$(obj)

#include $(DIR)/MAKE/simple_make.mk
include $(DIR)/MAKE/std_make.mk

OBJGQS = \
$(patsubst %,$(srcDIR)/%,$(srcOBJGQS)) \
$(patsubst %,$(prjDIR)/%,$(prjOBJGQS))





.f.o: 
	$(FRULE)

.cpp.o: 
	$(CRULE)

$(dim)d$(name).out: $(OBJGQS)
	$(ORULE)
	rm $(obj)

#if any header file is changed, the project file gets recompiled.
$(obj): $(StandardDependencies)


clean:
	\rm $(OBJGQS)



