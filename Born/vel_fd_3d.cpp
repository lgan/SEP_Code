#include "vel_fd_3d.h"
#include <math.h>
vel_fd_3d::vel_fd_3d(char *tag){
  tag_init(tag);
  fld=new hypercube_float(this);
  read_all(tag,fld);

  dens=0;
  old=false;
  mid=false;
  zero=false;
}

vel_fd_3d::vel_fd_3d(char *t1,char *t2){
  tag_init(t1);
  fld=new hypercube_float(this);
  read_all(t1,fld);
    dfile=new sregf(t2);
  dens=new hypercube_float(dfile);
  dfile->read_all(t2,dens);
  old=false;
  zero=false;
  mid=false;
}

hypercube_float *vel_fd_3d::rand_subspace(int irand, int nrand1, int nrand2, int nrand3, float dt, hypercube_float *space){
  float vmax=max_vel();
    srand(irand);
    float dy2,dx2,dz2,rnd;
    std::vector<axis> axes; axes.push_back(space->get_axis(1));
    axes.push_back(space->get_axis(2)); axes.push_back(space->get_axis(3));

    hypercube_float *tmp=new hypercube_float(axes);

    axis av_1=fld->get_axis(1);
    axis av_2=fld->get_axis(2);
    axis av_3=fld->get_axis(3);
    float *temp=new float[axes[0].n*axes[1].n*axes[2].n];
    int i=0;

    for(int iz=0; iz < axes[2].n; iz++){
      int iz_v=MAX(0,MIN(av_3.n-1,(int)((axes[2].o+axes[2].d*iz-av_3.o)/av_3.d)));
      if(iz < nrand3) dz2=(float)((nrand3-iz)*(nrand3-iz))/(float)(nrand3*nrand3);
      else if(iz >= axes[2].n-nrand3) 
        dz2=(float)((iz-(axes[2].n-nrand3-1))*(iz-(axes[2].n-nrand3-1)))/(float)(nrand3*nrand3);
      else dz2=0;

      for(int iy=0; iy < axes[1].n; iy++){
        int iy_v=MAX(0,MIN(av_2.n-1,(int)((axes[1].o+axes[1].d*iy-av_2.o)/av_2.d)));
        if(iy < nrand2) dy2=(float)((nrand2-iy)*(nrand2-iy))/(float)(nrand2*nrand2);
        else if(iy >= axes[1].n-nrand2) 
          dy2=(float)((iy-(axes[1].n-nrand2-1))*(iy-(axes[1].n-nrand2-1)))/(float)(nrand2*nrand2);
        else dy2=0;

        for(int ix=0; ix < axes[0].n; ix++,i++){ // NOTE the i++
          int ix_v=MAX(0,MIN(av_1.n-1,(int)((axes[0].o+axes[0].d*ix-av_1.o)/av_1.d)));
          float val=fld->vals[ix_v+iy_v*av_1.n+iz_v*av_1.n*av_2.n];
          if(ix < nrand1) dx2=(float)((nrand1-ix)*(nrand1-ix))/(float)(nrand1*nrand1);
          else if(ix >= axes[0].n-nrand1) 
            dx2=(float)((ix-(axes[0].n-nrand1-1))*(ix-(axes[0].n-nrand1-1)))/(float)(nrand1*nrand1);
          else dx2=0;
//fprintf(stderr,"%d x",ix);
	  if(dx2<0. || dx2>1 ) { fprintf(stderr,"Vel problem! %f %f %f, %d %d %d; %d %d\n",dx2,dy2,dz2,ix,iy,iz,nrand1,axes[0].n); dx2=1; }
          float dist=sqrtf(dx2+dy2+dz2);
          bool found=false;
          float dev;
          if(dist<.0001) found=true;
          else{
          //  val=val*1.03*(1.-MIN(1.,dist)*.83);
          }
          dev=0.;

          while(!found && nrand2!=0){
            rnd=(float)rand()/RAND_MAX-.5-.49*dist;
            dev=rnd*1.3*dist*val; //4.6 is arbitrary;
              if(fabs(dev+val) < vmax*1.03*(1.-MIN(1.,dist)*.4) && dev+val >0.0001) found=true;
          }


   temp[i]=val+dev;
   tmp->vals[i]=(val+dev)*(val+dev)*dt*dt;
        }
      }
    }

    delete []temp;
    return tmp;
}
