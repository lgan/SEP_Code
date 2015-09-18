#include "fd_prop_3d.h"
#include "sinc_bob.h"
#include "gpu_funcs_3d.h"
#include <math.h>


bool  fd_prop::calc_stability(std::vector<float> *ds,int n){

  vmax=vel->max_vel();
  float vmin=vel->min_vel();
  float dmin=vel->get_min_samp();
  float dmax=vel->get_max_samp();
  float d=.5*dmin/vmax;
  dt=ds->at(0)/ceilf(ds->at(0)/d);
  fprintf(stderr,"Minimum dt =%f %f %f\n",d,dt,ceilf(ds->at(0)/d));
  nt=(n-1)*(int)(ds->at(0)/dt)+1;
  for(int i=1; i < (int)ds->size(); i++) d=ds->at(i)/dt-(int)(ds->at(i)/dt+.001);

  if(fabs(d) >.01) {
    fprintf(stderr,"sampling match problem %f %f \n",d,dt);
    seperr("");
  }
  fprintf(stderr,"Stability check vmax=%f d=%f dt=%f fmax=%f \n",vmax,dmin,dt,vmin/2.8/dmax);
  return true;

}

void fd_prop::create_transfer_sinc_source(int nsinc){
  jts=(int)((source->get_dt())/dt);
  sinc_bob mys(jts,nsinc);
  transfer_sinc_table_s(nsinc,jts,mys.table);
}

void fd_prop::set_fd_basics(param_func *par,source_func *source_func, float ap,bool v){
  set_verb(v);
  aper=ap;
  float bc_a=par->get_float("bc_a",50.);
  float bc_b=par->get_float("bc_b",.0005);
  float bc_b_y=bc_b;
  source=source_func;
  nbound=par->get_int("nbound",(int)bc_a);
  nboundt=par->get_int("nboundt",nbound);
  int fat=4;
  int blocksize=16;
  create_transfer_sinc_source(8);

  source->set_compute_size(vel,aper,nbound,nbound,nbound,fat,blocksize);
  nx=source->x_points();
  nz=source->z_points();
  ny=source->y_points();
  create_gpu_space(vel->get_axis(1).d,vel->get_axis(2).d,vel->get_axis(3).d,bc_a,bc_b,bc_b_y,nx,ny,nz);

}
