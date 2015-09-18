#include "source_func_3d.h"
#include<math.h>

source_func::source_func(char *tag){

   tag_init(tag);
   hypercube_float *tmp=new hypercube_float(this);
   wavelet=new hypercube_float(this);
   read_all(tag,tmp);
   wavelet->vals[get_axis(1).n-1]=0;

   for(int it=0; it < this->get_axis(1).n-1; it++) {
     wavelet->vals[it]=(tmp->vals[it+1]-tmp->vals[it])/tmp->get_axis(1).d;
   }

   delete tmp;
   dt=get_axis(1).d;

}

void source_func::set_sources_3d(float s_z, axis src_axis1, axis src_axis2){

  for(int i2=0; i2 < src_axis2.n; i2++){
    for(int i1=0; i1 < src_axis1.n; i1++){
      sz.push_back(s_z);
      sx.push_back(src_axis1.o+src_axis1.d*i1);
      sy.push_back(src_axis2.o+src_axis2.d*i2);
    }
  }

}

int source_func::get_points(bool e ){
 return 9;
}

void source_func::set_compute_size(hypercube *dom, float ap,int nbt, int nb, int nby, int fat,
  int blocksize){
  ax=dom->get_axis(1);
  ay=dom->get_axis(2);
  az=dom->get_axis(3); 

  nbound=nb+fat; nboundt=nbt; nbound_y=nby+fat;
  aper=ap;

  ay.n+=2*nbound_y;
  ax.n+=2*nbound;
  az.n+=2*nbound;//nbt+nb+2*fat;
  az.n+=nbt+nb+2*fat;

  int rem_x=ax.n-((int) (ax.n/16))*16;
  int rem_y=ay.n-((int) (ay.n/16))*16;
  int rem_z=az.n-((int) (az.n/16))*16;

  if(rem_y!=0) rem_y=16-rem_y;
  if(rem_z!=0) rem_z=16-rem_z;
  if(rem_x!=0) rem_x=16-rem_x;

  //fprintf(stderr,"  Remainders are %d %d %d from %d %d %d. Padding z with %d + %d ; %d\n",rem_x,rem_y,rem_z,ax.n,ay.n,az.n,nbt,nb,nboundt);

  ay.n+=rem_y;
  ax.n+=rem_x;
  az.n+=rem_z;
//fprintf(stderr," Set comp %d %d %d , %f %f %f\n",ay.n,ax.n,az.n,ay.o,ax.o,az.o);
 /* ay.n=(int)(ap/ay.d)*2+1+2*nbound_y;
  ax.n=(int)(ap/ax.d)*2+1+2*nbound;
  az.n=az.n+nbt+nb+fat*2;
fprintf(stderr," Set comp %d %d %d , %f %f %f\n",ay.n,ax.n,az.n,ay.o,ax.o,az.o);

  int t1=(ax.n-fat*2)/blocksize;
  int rem=ax.n-t1*blocksize-2*fat;
  if(rem!=0) rem=blocksize-rem;
  if(rem!=0) ax.n+=+rem;

  t1=(ay.n-fat*2)/blocksize;
  rem=ay.n-t1*blocksize-2*fat;
  if(rem!=0) rem=blocksize-rem;
  if(rem!=0) ay.n+=+rem;
    
  t1=(az.n-fat*2)/blocksize;
  rem=az.n-t1*blocksize-2*fat;
  if(rem!=0) rem=blocksize-rem;
  if(rem!=0) az.n+=+rem;
fprintf(stderr," Set comp %d %d %d , %f %f %f\n",ay.n,ax.n,az.n,ay.o,ax.o,az.o);
*/

//  az.o=-20.;//az.o;//+(nbt+fat)*az.d;
}

//hypercube_float *source_func::create_domain(int ishotx, int ishoty){
hypercube_float *source_func::create_domain(int ishot){
  az.o=-az.d*nbound;
  ax.o=-ax.d*nbound;
  ay.o=-ay.d*nbound;
  std::vector<axis>  axes; axes.push_back(ax); axes.push_back(ay); axes.push_back(az);
  hypercube_float *tmp=new hypercube_float(axes);
  return tmp; 
}

void source_func::get_source_func(hypercube_float *domain, int ishot, int nts, int *locs, float *time){

  axis a1=domain->get_axis(1);
  axis a2=domain->get_axis(2);
  axis a3=domain->get_axis(3);
  axis at=get_axis(1);

  float fz=(sz[ishot]-a3.o)/a3.d;
  float fy=(sy[ishot]-a2.o)/a2.d;
  float fx=(sx[ishot]-a1.o)/a1.d;
  int iy=(int)fy;
  int ix=(int)fx;
  int iz=(int)fz;
  int i=0;

  for(int it=0; it < nts*9; it++){
    time[it]=0;
  }
  for(int i3=iz-1; i3 <= iz-1; i3++){
    float zn=fz-i3;
    for(int i2=iy-1; i2 <= iy+1; i2++){
      float yn=fy-i2;
      for(int i1=ix-1; i1<= ix+1; i1++,i++){
        float xn=fx-i1;
        locs[i]=i1+i2*a1.n+i3*a1.n*a2.n;
        float scale=expf(-zn*zn-xn*xn-yn*yn);
        for(int it=0; it < at.n; it++){
          time[nts*i+it+4]=scale*wavelet->vals[it];
        }
      }
    }
  }

}


