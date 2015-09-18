#!/usr/bin/env python
import os
import sys
import commands
import random

data = sys.argv[1]
nshots_x = int(sys.argv[2])
nshots_y = int(sys.argv[3])
nblend = int(sys.argv[4])
total_time = int(sys.argv[5])
shift_range = int(sys.argv[6])
fileout = sys.argv[7]
#sx=0
sy=0
nb=nshots_x-nblend
ob=0
os=0
sx=0
outs=" "
#rand_array=[]

#while sx < nshots_x:
for sb in range(0,nb):
  tmpfile = "tmp.b.%s.H"%(sb)
  print "Blend shots in groups of %d"%(nblend)
  os=0
  out_t=" "
  for sx in range(ob,ob+nblend):
    print "  Blend %d to %d"%(ob,ob+nblend)
    rand_number = random.randrange(1,shift_range+1)
    #shot_x = random.randrange(0,nshots_x+1)
  #while sy < nshots_y:
  #for sy in range(0,nshots_y):
    if sx==ob:
      rand_number=0
    shot_x=sx
    filename = "tmp.r.%d.%d.H"%(sx,sb)
    #shift = rand_number + 100*sy
    #shift = (shift_range / nblend)*os
    shift = rand_number
    print "Window3d n4=1 n5=1 f4=%d f5=%d < %s | Pad beg1=%d | Window3d n1=%d > %s"%(shot_x,sy,data,shift,total_time,filename)
    commands.getstatusoutput("Window3d n4=1 n5=1 f4=%d f5=%d < %s | Pad beg1=%d | Window3d n1=%d > %s"%(shot_x,sy,data,shift,total_time,filename))
    commands.getstatusoutput("echo \"o1=0. o4=0. o5=0.\" >> %s"%(filename))

    #if sx==0:
    #  if sy==0:
    #    commands.getstatusoutput("cp %s %s"%(filename,fileout))
    #    print "cp %s %s"%(filename,fileout)
    #  else:
    #    commands.getstatusoutput("cp %s tmp.r.total.H"%(fileout))
    #    print "cp %s tmp.r.total.H"%(fileout)
    #    commands.getstatusoutput("Add mode=sum file1=tmp.r.total.H file2=%s > %s"%(filename,fileout))
    #    print "Add mode=sum tmp.r.total.H %s > %s"%(filename,fileout)
    #else:
    #  commands.getstatusoutput("cp %s tmp.r.total.H"%(fileout))
    #  print "cp %s tmp.r.total.H"%(fileout)
    #  commands.getstatusoutput("Add mode=sum file1=tmp.r.total.H file2=%s > %s"%(filename,fileout))
    #  print "Add mode=sum tmp.r.total.H %s > %s"%(filename,fileout)

    #sy = sy + 1
    sx = sx + 1
    os = os + 1
    out_t += " %s"%(filename)
  print "Add mode=sum %s > %s"%(out_t,tmpfile)
  commands.getstatusoutput("Add mode=sum %s > %s"%(out_t,tmpfile))
  ob = ob+1
  outs += " %s"%(tmpfile)
  commands.getstatusoutput("Rm tmp.r.*")
#commands.getstatusoutput("Add mode=sum tmp.l.* > %s"%(fileout))
print "Cat3d %s axis=4 > %s"%(outs,fileout)
commands.getstatusoutput("Cat3d %s axis=4 > %s"%(outs,fileout))
commands.getstatusoutput("Rm tmp.b.*")

