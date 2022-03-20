import sys,os
sys.path.append("/home/ridout/jscripts")
import res
import ncwrapper as nc
import numpy as np
import subprocess
import glob
import re
import time
step= int(sys.argv[1])

m = 1
M= 65
pydir = "/home1/ridout/packages/ML_softness/"
jsrcdir = "/home1/ridout/jsrc/"
jsubfunc = res.simple_marmalade_job
shortq="compute"
with open("/home1/ridout/packages/lib_softness/notebooks/df_targets.txt", "r") as f:
    for basename in f: 
        basedir = basename.rstrip()
        try:
            if time.time() - os.path.getmtime(basedir+str(m)+"/state.nc") < 28*24*60*60 :
                print((time.time() - os.path.getmtime(basedir+"1/state.nc")) / (24*60*60))
            else:
                continue
        except:
            continue
        try:
          N = re.search("N=[0123456789]*_", basedir)[0][2:-1]
          dim = re.search("[0123456789]*dshear",basedir)[0][:-6]
          print("N=",N)
          print("d=",dim)
        except:
          continue
        for i in range(m,M):
            if step == 0:
                #if not os.path.isfile(basedir+str(i)+".gsd"):
                    jobstring = "python " + pydir +"A0_nc_to_gsd.py " +basedir+str(i)+"/state.nc" 
                    print(jobstring) 
                    jsubfunc(jobstring=jobstring,jobname="A0_"+str(i),time="2",q=shortq) 
            if step == 2:
                dirname = basedir+str(i)+"/"
                jsubfunc(jobstring="python "+pydir+"record_drops.py " + dirname, jobname="drops",time="1",q=shortq)
            if step == 3:
             try:
               split = 50
               def file_len(fname):
                   with open(fname) as f:
                       for i, l in enumerate(f):
                           pass
                   return i + 1 
               total = file_len(basedir+str(i)+"/drops.txt")
               if "Lp" in basedir:
                  jobstring = "source /home/ridout/jsrc/mains/jload \n"+jsrcdir+"mains/"+dim+"denthSamModeD2min.out "
                  os.makedirs(basedir+str(i)+"/lowmodesEnth",exist_ok=True)
                  os.makedirs(basedir+str(i)+"/moded2minEnth",exist_ok=True)
               else:
                  jobstring = "source /home/ridout/jsrc/mains/jload \n"+jsrcdir+"mains/"+dim+"dsamModeD2min.out "    

               os.makedirs(basedir+str(i)+"/moded2min2",exist_ok=True)
               os.makedirs(basedir+str(i)+"/lowmodes2",exist_ok=True)
               for j in range(int(np.ceil(total/split))):
                  full_jobstring = jobstring + basedir+str(i) + " " + N + " " + str(j*split) + " " +  str((j+1)*split) 
                  jsubfunc(jobstring=full_jobstring,jobname="modes_"+str(i),time="24",q=shortq)
             except Exception as e:
                print(e)
            
            if step == 5:
                script = ""
                if dim == "4" or dim == "5":
                   script += "4d"
                script += "A1-part2-drops"
                if "Lp" in basedir:
                    script += "H"
                script += ".py"
                jobstring = "time python " + pydir +script + " "+ basedir+str(i)+".gsd perc"

                jsubfunc(jobstring=jobstring,jobname="dropdeco_"+str(i),q=shortq,time="5",dep="2234352:2234360")  

            if step == 6:
             try:
               split = 1000
               def file_len(fname):
                   with open(fname) as f:
                       for i, l in enumerate(f):
                           pass
                   return i + 1
               total = file_len(basedir+str(i)+"/drops.txt")
               jobstring = "source /home1/ridout/jsrc/mains/jload \n"+jsrcdir+"mains/"+dim+"ddumpRattlerMap.out "

               os.makedirs(basedir+str(i)+"/rattlermaps",exist_ok=True)
               for j in range(int(np.ceil(total/split))):
                  full_jobstring = jobstring + basedir+str(i) + " " + N + " " + str(j*split) + " " +  str((j+1)*split)
                  jsubfunc(jobstring=full_jobstring,jobname="ratlers_"+str(i),time="24",q=shortq)
             except Exception as e:
                print(e)
