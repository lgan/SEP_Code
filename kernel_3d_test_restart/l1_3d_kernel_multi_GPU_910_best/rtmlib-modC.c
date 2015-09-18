/***********************************************************************
* Filename : rtmlib-modC.c
* Create : Lin 2015-07-13
* Description:This is the C code to implement the same functions in ./rtmlib-modF.f90 
* Modified   : 
* Revision of last commit: $Rev$ 
* Author of last commit  : $Author$ $date$
* licence :
$licence$
* **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "gpu.h"

#include "sep3d.h"
#include "seplib.h"

void func_check(int ny, int nx, int nz, float *a, float *b){
	int iy, iz, ix;
	int flag = 1;
	for(iy = 0; iy < ny && flag; iy++)
		for(ix = 0; ix < nx && flag; ix++)
			for(iz = 0; iz < nz && flag; iz++){
				if(fabs(a[n3d_index_ex(iz,ix, iy)] - b[n3d_index_ex(iz,ix, iy)]) > 10e-14){
//					fprintf(stderr, "%d,%d,%d, %.14f\n",iy,ix, iz,   0);// a[n3d_index_ex(iz,ix, iy)] );
//					fprintf(stderr, "%d,%d,%d, %.14f\n\n",iy, ix, iz,0);// b[n3d_index_ex(iz,ix, iy)] );
					flag=0;
					//break;	
					}
			}
	if(flag) fprintf(stderr, "Check Okay\n");
	else	fprintf(stderr, "Check Error\n");
			
}	


//gause source function
void source_op2(float *source, int nt){
	
	float dt = 0.4*(4.0)/(sqrt(2.0)*2000.0);
	int nsource = 250;
	int ft = nsource/10;
	float pi = 3.14159265;
	float t;
	float aux;
	float fp = 20.0;

	for(int it=0; it<nt; it++){
	if(it>=ft){
		t = (it*dt) - (nsource*dt/2.0);
		aux = (pi*pi*fp*fp*t*t);
		source[it] = (10.0e7)*(1.0 -2.0*aux)*exp(-aux);
		}
	else{
		source[it] = 0.0;
	    }
	}
}



//add source
void add_source(float source, int ysource, int xsource, int zsource, int ny, int nx, int nz, float *data){

	float dt = 0.4*(4.0)/(sqrt(2.0)*2000.0);
	data[n3d_index_ex(zsource, xsource, ysource)]+=dt*source;	
}
	
	
void model_op(float *ex_m1_y, float *ex_m1_x, float *ex_m1_z, float *ex_m2, float *ex_m3, float *ex_m2m3, int ny, int nx, int nz){

	float rho=1000.0, lambda=2.25e9, mu=0.0;
	float m2 = lambda;
	float m3 = mu;
	float dt = 0.4*(4.0)/(sqrt(2.0)*2000.0);
	float dx = 4.0;
	float aux = dt/dx;

	//linear interpolation for mx,my,nz
	//This loop copied from model_op() in model-mode.f90 and the rtmmod_init() from rtmlib-mod.f90 in 2D elastic code in 2D elastic code
	for(int iy=0; iy<ny; iy++)
		for(int ix=0; ix<nx; ix++)
			for(int iz=0; iz<nz; iz++){
				ex_m1_x[n3d_index_ex(iz,ix,iy)] = (aux)*(1.0/(rho));
				ex_m1_y[n3d_index_ex(iz,ix,iy)] = (aux)*(1.0/(rho));
				ex_m1_z[n3d_index_ex(iz,ix,iy)] = (aux)*(1.0/(rho));
				ex_m2[n3d_index_ex(iz,ix,iy)] = aux*m2;
				ex_m3[n3d_index_ex(iz,ix,iy)] = aux*m3;
				ex_m2m3[n3d_index_ex(iz,ix,iy)] = aux*(m2+2*m3);
		}

}



void rtm3d_op(int ny, int nz, int nx,
	float *ex_Vy0_now, float *ex_Vx0_now, float *ex_Vz0_now, float *ex_sigmayy0_now, float *ex_sigmaxx0_now, float *ex_sigmazz0_now, float *ex_sigmaxy0_now, float *ex_sigmaxz0_now, float *ex_sigmayz0_now,
	float *ex_Vy0_next, float *ex_Vx0_next, float *ex_Vz0_next, float *ex_sigmayy0_next, float *ex_sigmaxx0_next, float *ex_sigmazz0_next, float *ex_sigmaxy0_next, float *ex_sigmaxz0_next, float *ex_sigmayz0_next,
	float *ex_Vy0_pre, float *ex_Vx0_pre, float *ex_Vz0_pre, float *ex_sigmayy0_pre, float *ex_sigmaxx0_pre, float *ex_sigmazz0_pre, float *ex_sigmaxy0_pre, float *ex_sigmaxz0_pre, float *ex_sigmayz0_pre,
	float *ex_m1_y, float *ex_m1_x, float *ex_m1_z, float *ex_m2, float *ex_m3, float *ex_m2m3){
	
   	float c1=35.0/294912.0,c2=-405.0/229376.0,c3=567.0/40960.0,c4=-735.0/8192.0,c5=19845.0/16384.0;

	int iy, iz, ix;

omp_set_num_threads(NUM_OMP_THREADS);
	
#pragma omp parallel //for private(iy,ix,iz) 
{
	if(omp_get_thread_num() == 0){
		fprintf(stderr, "Using %d CPU threads (OpenMP)\n", omp_get_num_threads());
	}

#pragma omp for private(iy,ix,iz) 
  	for(iy=0;iy<ny;iy++)
            for(ix=0; ix<nx; ix++)
                for(iz=0; iz<nz; iz++){
 
 		         	ex_Vx0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_Vx0_pre[n3d_index_ex(iz,ix  ,iy)]	+*/ ex_Vx0_next[n3d_index_ex(iz, ix, iy)]	

									+ ex_m2m3[n3d_index_ex(iz,ix-5, iy)]*c1*ex_sigmaxx0_now[n3d_index_ex(iz,ix-5,iy)]							
							 		+ ex_m2m3[n3d_index_ex(iz,ix-4, iy)]*c2*ex_sigmaxx0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ ex_m2m3[n3d_index_ex(iz,ix-3, iy)]*c3*ex_sigmaxx0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ ex_m2m3[n3d_index_ex(iz,ix-2, iy)]*c4*ex_sigmaxx0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ ex_m2m3[n3d_index_ex(iz,ix-1, iy)]*c5*ex_sigmaxx0_now[n3d_index_ex(iz,ix-1,iy)]	
									- ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m2m3[n3d_index_ex(iz,ix+1, iy)]*c4*ex_sigmaxx0_now[n3d_index_ex(iz,ix+1,iy)]	
									- ex_m2m3[n3d_index_ex(iz,ix+2, iy)]*c3*ex_sigmaxx0_now[n3d_index_ex(iz,ix+2,iy)]	
									- ex_m2m3[n3d_index_ex(iz,ix+3, iy)]*c2*ex_sigmaxx0_now[n3d_index_ex(iz,ix+3,iy)]	
									- ex_m2m3[n3d_index_ex(iz,ix+4, iy)]*c1*ex_sigmaxx0_now[n3d_index_ex(iz,ix+4,iy)]
	

									+ ex_m2[n3d_index_ex(iz,ix-5, iy)]*c1*ex_sigmayy0_now[n3d_index_ex(iz,ix-5,iy)]							
							 		+ ex_m2[n3d_index_ex(iz,ix-4, iy)]*c2*ex_sigmayy0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ ex_m2[n3d_index_ex(iz,ix-3, iy)]*c3*ex_sigmayy0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ ex_m2[n3d_index_ex(iz,ix-2, iy)]*c4*ex_sigmayy0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ ex_m2[n3d_index_ex(iz,ix-1, iy)]*c5*ex_sigmayy0_now[n3d_index_ex(iz,ix-1,iy)]	
									- ex_m2[n3d_index_ex(iz,  ix, iy)]*c5*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+1, iy)]*c4*ex_sigmayy0_now[n3d_index_ex(iz,ix+1,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+2, iy)]*c3*ex_sigmayy0_now[n3d_index_ex(iz,ix+2,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+3, iy)]*c2*ex_sigmayy0_now[n3d_index_ex(iz,ix+3,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+4, iy)]*c1*ex_sigmayy0_now[n3d_index_ex(iz,ix+4,iy)]	
	

									+ ex_m2[n3d_index_ex(iz,ix-5, iy)]*c1*ex_sigmazz0_now[n3d_index_ex(iz,ix-5,iy)]							
							 		+ ex_m2[n3d_index_ex(iz,ix-4, iy)]*c2*ex_sigmazz0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ ex_m2[n3d_index_ex(iz,ix-3, iy)]*c3*ex_sigmazz0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ ex_m2[n3d_index_ex(iz,ix-2, iy)]*c4*ex_sigmazz0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ ex_m2[n3d_index_ex(iz,ix-1, iy)]*c5*ex_sigmazz0_now[n3d_index_ex(iz,ix-1,iy)]	
									- ex_m2[n3d_index_ex(iz,  ix, iy)]*c5*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+1, iy)]*c4*ex_sigmazz0_now[n3d_index_ex(iz,ix+1,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+2, iy)]*c3*ex_sigmazz0_now[n3d_index_ex(iz,ix+2,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+3, iy)]*c2*ex_sigmazz0_now[n3d_index_ex(iz,ix+3,iy)]	
									- ex_m2[n3d_index_ex(iz,ix+4, iy)]*c1*ex_sigmazz0_now[n3d_index_ex(iz,ix+4,iy)]	
	

									+ ex_m3[n3d_index_ex(iz,ix, iy-4)]*c1*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ ex_m3[n3d_index_ex(iz,ix, iy-3)]*c2*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ ex_m3[n3d_index_ex(iz,ix, iy-2)]*c3*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ ex_m3[n3d_index_ex(iz,ix, iy-1)]*c4*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m3[n3d_index_ex(iz,ix, iy+1)]*c5*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+1)]	
									- ex_m3[n3d_index_ex(iz,ix, iy+2)]*c4*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+2)]	
									- ex_m3[n3d_index_ex(iz,ix, iy+3)]*c3*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+3)]	
									- ex_m3[n3d_index_ex(iz,ix, iy+4)]*c2*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+4)]	
									- ex_m3[n3d_index_ex(iz,ix, iy+5)]*c1*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+5)]							
	

									+ ex_m3[n3d_index_ex(iz-4,ix, iy)]*c1*ex_sigmaxz0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ ex_m3[n3d_index_ex(iz-3,ix, iy)]*c2*ex_sigmaxz0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ ex_m3[n3d_index_ex(iz-2,ix, iy)]*c3*ex_sigmaxz0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ ex_m3[n3d_index_ex(iz-1,ix, iy)]*c4*ex_sigmaxz0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ ex_m3[n3d_index_ex(iz,  ix, iy)]*c5*ex_sigmaxz0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m3[n3d_index_ex(iz+1,ix, iy)]*c5*ex_sigmaxz0_now[n3d_index_ex(iz+1,ix,iy)]	
									- ex_m3[n3d_index_ex(iz+2,ix, iy)]*c4*ex_sigmaxz0_now[n3d_index_ex(iz+2,ix,iy)]	
									- ex_m3[n3d_index_ex(iz+3,ix, iy)]*c3*ex_sigmaxz0_now[n3d_index_ex(iz+3,ix,iy)]	
									- ex_m3[n3d_index_ex(iz+4,ix, iy)]*c2*ex_sigmaxz0_now[n3d_index_ex(iz+4,ix,iy)]	
									- ex_m3[n3d_index_ex(iz+5,ix, iy)]*c1*ex_sigmaxz0_now[n3d_index_ex(iz+5,ix,iy)]	;						
	

        	ex_Vy0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_Vy0_pre[n3d_index_ex(iz,ix  ,iy)]*/	+ ex_Vy0_next[n3d_index_ex(iz, ix, iy)]	

     								+ ex_m2m3[n3d_index_ex(iz,ix, iy-5)]*c1*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-5)]							
     						 		+ ex_m2m3[n3d_index_ex(iz,ix, iy-4)]*c2*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-4)]		
     								+ ex_m2m3[n3d_index_ex(iz,ix, iy-3)]*c3*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-3)]	
     								+ ex_m2m3[n3d_index_ex(iz,ix, iy-2)]*c4*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-2)]	
     								+ ex_m2m3[n3d_index_ex(iz,ix, iy-1)]*c5*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-1)]	
     								- ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m2m3[n3d_index_ex(iz,ix, iy+1)]*c4*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+1)]	
     								- ex_m2m3[n3d_index_ex(iz,ix, iy+2)]*c3*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+2)]	
     								- ex_m2m3[n3d_index_ex(iz,ix, iy+3)]*c2*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+3)]	
     								- ex_m2m3[n3d_index_ex(iz,ix, iy+4)]*c1*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+4)]
     

     								+ ex_m2[n3d_index_ex(iz,ix, iy-5)]*c1*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-5)]							
     						 		+ ex_m2[n3d_index_ex(iz,ix, iy-4)]*c2*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-4)]		
     								+ ex_m2[n3d_index_ex(iz,ix, iy-3)]*c3*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-3)]	
     								+ ex_m2[n3d_index_ex(iz,ix, iy-2)]*c4*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-2)]	
     								+ ex_m2[n3d_index_ex(iz,ix, iy-1)]*c5*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-1)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+1)]*c4*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+1)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+2)]*c3*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+2)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+3)]*c2*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+3)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+4)]*c1*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+4)]	
     

     								+ ex_m2[n3d_index_ex(iz,ix, iy-5)]*c1*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-5)]							
     						 		+ ex_m2[n3d_index_ex(iz,ix, iy-4)]*c2*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-4)]		
     								+ ex_m2[n3d_index_ex(iz,ix, iy-3)]*c3*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-3)]	
     								+ ex_m2[n3d_index_ex(iz,ix, iy-2)]*c4*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-2)]	
     								+ ex_m2[n3d_index_ex(iz,ix, iy-1)]*c5*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-1)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+1)]*c4*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+1)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+2)]*c3*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+2)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+3)]*c2*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+3)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy+4)]*c1*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+4)]	
     

     								+ ex_m3[n3d_index_ex(iz-4,ix, iy)]*c1*ex_sigmayz0_now[n3d_index_ex(iz-4,ix,iy)]		
     								+ ex_m3[n3d_index_ex(iz-3,ix, iy)]*c2*ex_sigmayz0_now[n3d_index_ex(iz-3,ix,iy)]	
     								+ ex_m3[n3d_index_ex(iz-2,ix, iy)]*c3*ex_sigmayz0_now[n3d_index_ex(iz-2,ix,iy)]	
     								+ ex_m3[n3d_index_ex(iz-1,ix, iy)]*c4*ex_sigmayz0_now[n3d_index_ex(iz-1,ix,iy)]	
     								+ ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz+1,ix, iy)]*c5*ex_sigmayz0_now[n3d_index_ex(iz+1,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz+2,ix, iy)]*c4*ex_sigmayz0_now[n3d_index_ex(iz+2,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz+3,ix, iy)]*c3*ex_sigmayz0_now[n3d_index_ex(iz+3,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz+4,ix, iy)]*c2*ex_sigmayz0_now[n3d_index_ex(iz+4,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz+5,ix, iy)]*c1*ex_sigmayz0_now[n3d_index_ex(iz+5,ix,iy)]							
     

     								+ ex_m3[n3d_index_ex(iz,ix-4, iy)]*c1*ex_sigmaxy0_now[n3d_index_ex(iz,ix-4,iy)]		
     								+ ex_m3[n3d_index_ex(iz,ix-3, iy)]*c2*ex_sigmaxy0_now[n3d_index_ex(iz,ix-3,iy)]	
     								+ ex_m3[n3d_index_ex(iz,ix-2, iy)]*c3*ex_sigmaxy0_now[n3d_index_ex(iz,ix-2,iy)]	
     								+ ex_m3[n3d_index_ex(iz,ix-1, iy)]*c4*ex_sigmaxy0_now[n3d_index_ex(iz,ix-1,iy)]	
     								+ ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+1, iy)]*c5*ex_sigmaxy0_now[n3d_index_ex(iz,ix+1,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+2, iy)]*c4*ex_sigmaxy0_now[n3d_index_ex(iz,ix+2,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+3, iy)]*c3*ex_sigmaxy0_now[n3d_index_ex(iz,ix+3,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+4, iy)]*c2*ex_sigmaxy0_now[n3d_index_ex(iz,ix+4,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+5, iy)]*c1*ex_sigmaxy0_now[n3d_index_ex(iz,ix+5,iy)]	;						




        	ex_Vz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_Vz0_pre[n3d_index_ex(iz,ix  ,iy)]*/	+ ex_Vz0_next[n3d_index_ex(iz, ix, iy)]	

     								+ ex_m2m3[n3d_index_ex(iz-5,ix, iy)]*c1*ex_sigmazz0_now[n3d_index_ex(iz-5,ix,iy)]							
     						 		+ ex_m2m3[n3d_index_ex(iz-4,ix, iy)]*c2*ex_sigmazz0_now[n3d_index_ex(iz-4,ix,iy)]		
     								+ ex_m2m3[n3d_index_ex(iz-3,ix, iy)]*c3*ex_sigmazz0_now[n3d_index_ex(iz-3,ix,iy)]	
     								+ ex_m2m3[n3d_index_ex(iz-2,ix, iy)]*c4*ex_sigmazz0_now[n3d_index_ex(iz-2,ix,iy)]	
     								+ ex_m2m3[n3d_index_ex(iz-1,ix, iy)]*c5*ex_sigmazz0_now[n3d_index_ex(iz-1,ix,iy)]	
     								- ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmazz0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m2m3[n3d_index_ex(iz+1,ix, iy)]*c4*ex_sigmazz0_now[n3d_index_ex(iz+1,ix,iy)]	
     								- ex_m2m3[n3d_index_ex(iz+2,ix, iy)]*c3*ex_sigmazz0_now[n3d_index_ex(iz+2,ix,iy)]	
     								- ex_m2m3[n3d_index_ex(iz+3,ix, iy)]*c2*ex_sigmazz0_now[n3d_index_ex(iz+3,ix,iy)]	
     								- ex_m2m3[n3d_index_ex(iz+4,ix, iy)]*c1*ex_sigmazz0_now[n3d_index_ex(iz+4,ix,iy)]
     

     								+ ex_m2[n3d_index_ex(iz-5,ix, iy)]*c1*ex_sigmaxx0_now[n3d_index_ex(iz-5,ix,iy)]							
     						 		+ ex_m2[n3d_index_ex(iz-4,ix, iy)]*c2*ex_sigmaxx0_now[n3d_index_ex(iz-4,ix,iy)]		
     								+ ex_m2[n3d_index_ex(iz-3,ix, iy)]*c3*ex_sigmaxx0_now[n3d_index_ex(iz-3,ix,iy)]	
     								+ ex_m2[n3d_index_ex(iz-2,ix, iy)]*c4*ex_sigmaxx0_now[n3d_index_ex(iz-2,ix,iy)]	
     								+ ex_m2[n3d_index_ex(iz-1,ix, iy)]*c5*ex_sigmaxx0_now[n3d_index_ex(iz-1,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+1,ix, iy)]*c4*ex_sigmaxx0_now[n3d_index_ex(iz+1,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+2,ix, iy)]*c3*ex_sigmaxx0_now[n3d_index_ex(iz+2,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+3,ix, iy)]*c2*ex_sigmaxx0_now[n3d_index_ex(iz+3,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+4,ix, iy)]*c1*ex_sigmaxx0_now[n3d_index_ex(iz+4,ix,iy)]
     

     								+ ex_m2[n3d_index_ex(iz-5,ix, iy)]*c1*ex_sigmayy0_now[n3d_index_ex(iz-5,ix,iy)]							
     						 		+ ex_m2[n3d_index_ex(iz-4,ix, iy)]*c2*ex_sigmayy0_now[n3d_index_ex(iz-4,ix,iy)]		
     								+ ex_m2[n3d_index_ex(iz-3,ix, iy)]*c3*ex_sigmayy0_now[n3d_index_ex(iz-3,ix,iy)]	
     								+ ex_m2[n3d_index_ex(iz-2,ix, iy)]*c4*ex_sigmayy0_now[n3d_index_ex(iz-2,ix,iy)]	
     								+ ex_m2[n3d_index_ex(iz-1,ix, iy)]*c5*ex_sigmayy0_now[n3d_index_ex(iz-1,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmayy0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+1,ix, iy)]*c4*ex_sigmayy0_now[n3d_index_ex(iz+1,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+2,ix, iy)]*c3*ex_sigmayy0_now[n3d_index_ex(iz+2,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+3,ix, iy)]*c2*ex_sigmayy0_now[n3d_index_ex(iz+3,ix,iy)]	
     								- ex_m2[n3d_index_ex(iz+4,ix, iy)]*c1*ex_sigmayy0_now[n3d_index_ex(iz+4,ix,iy)]
     
     								+ ex_m3[n3d_index_ex(iz,ix, iy-4)]*c1*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-4)]		
     								+ ex_m3[n3d_index_ex(iz,ix, iy-3)]*c2*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-3)]	
     								+ ex_m3[n3d_index_ex(iz,ix, iy-2)]*c3*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-2)]	
     								+ ex_m3[n3d_index_ex(iz,ix, iy-1)]*c4*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-1)]	
     								+ ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix, iy+1)]*c5*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+1)]	
     								- ex_m3[n3d_index_ex(iz,ix, iy+2)]*c4*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+2)]	
     								- ex_m3[n3d_index_ex(iz,ix, iy+3)]*c3*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+3)]	
     								- ex_m3[n3d_index_ex(iz,ix, iy+4)]*c2*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+4)]	
     								- ex_m3[n3d_index_ex(iz,ix, iy+5)]*c1*ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+5)]							
     

     								+ ex_m3[n3d_index_ex(iz,ix-4, iy)]*c1*ex_sigmaxz0_now[n3d_index_ex(iz,ix-4,iy)]		
     								+ ex_m3[n3d_index_ex(iz,ix-3, iy)]*c2*ex_sigmaxz0_now[n3d_index_ex(iz,ix-3,iy)]	
     								+ ex_m3[n3d_index_ex(iz,ix-2, iy)]*c3*ex_sigmaxz0_now[n3d_index_ex(iz,ix-2,iy)]	
     								+ ex_m3[n3d_index_ex(iz,ix-1, iy)]*c4*ex_sigmaxz0_now[n3d_index_ex(iz,ix-1,iy)]	
     								+ ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*ex_sigmaxz0_now[n3d_index_ex(iz,ix,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+1, iy)]*c5*ex_sigmaxz0_now[n3d_index_ex(iz,ix+1,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+2, iy)]*c4*ex_sigmaxz0_now[n3d_index_ex(iz,ix+2,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+3, iy)]*c3*ex_sigmaxz0_now[n3d_index_ex(iz,ix+3,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+4, iy)]*c2*ex_sigmaxz0_now[n3d_index_ex(iz,ix+4,iy)]	
     								- ex_m3[n3d_index_ex(iz,ix+5, iy)]*c1*ex_sigmaxz0_now[n3d_index_ex(iz,ix+5,iy)]	;						


     	

              ex_sigmaxx0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_sigmaxx0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ ex_sigmaxx0_next[n3d_index_ex(iz,ix  , iy)] 
									+ ex_m1_x[n3d_index_ex(iz,ix-4, iy)]*c1*ex_Vx0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ ex_m1_x[n3d_index_ex(iz,ix-3, iy)]*c2*ex_Vx0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ ex_m1_x[n3d_index_ex(iz,ix-2, iy)]*c3*ex_Vx0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ ex_m1_x[n3d_index_ex(iz,ix-1, iy)]*c4*ex_Vx0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vx0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_x[n3d_index_ex(iz,ix+1, iy)]*c5*ex_Vx0_now[n3d_index_ex(iz,ix+1,iy)]	
									- ex_m1_x[n3d_index_ex(iz,ix+2, iy)]*c4*ex_Vx0_now[n3d_index_ex(iz,ix+2,iy)]	
									- ex_m1_x[n3d_index_ex(iz,ix+3, iy)]*c3*ex_Vx0_now[n3d_index_ex(iz,ix+3,iy)]	
									- ex_m1_x[n3d_index_ex(iz,ix+4, iy)]*c2*ex_Vx0_now[n3d_index_ex(iz,ix+4,iy)]	
									- ex_m1_x[n3d_index_ex(iz,ix+5, iy)]*c1*ex_Vx0_now[n3d_index_ex(iz,ix+5,iy)]	;						

	    
              ex_sigmayy0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_sigmayy0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ ex_sigmayy0_next[n3d_index_ex(iz,ix  , iy)] 
									+ ex_m1_y[n3d_index_ex(iz,ix, iy-4)]*c1*ex_Vy0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ ex_m1_y[n3d_index_ex(iz,ix, iy-3)]*c2*ex_Vy0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ ex_m1_y[n3d_index_ex(iz,ix, iy-2)]*c3*ex_Vy0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ ex_m1_y[n3d_index_ex(iz,ix, iy-1)]*c4*ex_Vy0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vy0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_y[n3d_index_ex(iz,ix, iy+1)]*c5*ex_Vy0_now[n3d_index_ex(iz,ix,iy+1)]	
									- ex_m1_y[n3d_index_ex(iz,ix, iy+2)]*c4*ex_Vy0_now[n3d_index_ex(iz,ix,iy+2)]	
									- ex_m1_y[n3d_index_ex(iz,ix, iy+3)]*c3*ex_Vy0_now[n3d_index_ex(iz,ix,iy+3)]	
									- ex_m1_y[n3d_index_ex(iz,ix, iy+4)]*c2*ex_Vy0_now[n3d_index_ex(iz,ix,iy+4)]	
									- ex_m1_y[n3d_index_ex(iz,ix, iy+5)]*c1*ex_Vy0_now[n3d_index_ex(iz,ix,iy+5)]	;		


              ex_sigmazz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_sigmazz0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ ex_sigmazz0_next[n3d_index_ex(iz,ix  , iy)] 
									+ ex_m1_z[n3d_index_ex(iz-4,ix, iy)]*c1*ex_Vz0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ ex_m1_z[n3d_index_ex(iz-3,ix, iy)]*c2*ex_Vz0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ ex_m1_z[n3d_index_ex(iz-2,ix, iy)]*c3*ex_Vz0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ ex_m1_z[n3d_index_ex(iz-1,ix, iy)]*c4*ex_Vz0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vz0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_z[n3d_index_ex(iz+1,ix, iy)]*c5*ex_Vz0_now[n3d_index_ex(iz+1,ix,iy)]	
									- ex_m1_z[n3d_index_ex(iz+2,ix, iy)]*c4*ex_Vz0_now[n3d_index_ex(iz+2,ix,iy)]	
									- ex_m1_z[n3d_index_ex(iz+3,ix, iy)]*c3*ex_Vz0_now[n3d_index_ex(iz+3,ix,iy)]	
									- ex_m1_z[n3d_index_ex(iz+4,ix, iy)]*c2*ex_Vz0_now[n3d_index_ex(iz+4,ix,iy)]	
									- ex_m1_z[n3d_index_ex(iz+5,ix, iy)]*c1*ex_Vz0_now[n3d_index_ex(iz+5,ix,iy)]	;		 
	
	


              ex_sigmaxy0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_sigmaxy0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ ex_sigmaxy0_next[n3d_index_ex(iz,ix  , iy)] 
									+ ex_m1_y[n3d_index_ex(iz,ix-4, iy)]*c1*ex_Vy0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ ex_m1_y[n3d_index_ex(iz,ix-3, iy)]*c2*ex_Vy0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ ex_m1_y[n3d_index_ex(iz,ix-2, iy)]*c3*ex_Vy0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ ex_m1_y[n3d_index_ex(iz,ix-1, iy)]*c4*ex_Vy0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vy0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_y[n3d_index_ex(iz,ix+1, iy)]*c5*ex_Vy0_now[n3d_index_ex(iz,ix+1,iy)]	
									- ex_m1_y[n3d_index_ex(iz,ix+2, iy)]*c4*ex_Vy0_now[n3d_index_ex(iz,ix+2,iy)]	
									- ex_m1_y[n3d_index_ex(iz,ix+3, iy)]*c3*ex_Vy0_now[n3d_index_ex(iz,ix+3,iy)]	
									- ex_m1_y[n3d_index_ex(iz,ix+4, iy)]*c2*ex_Vy0_now[n3d_index_ex(iz,ix+4,iy)]	
									- ex_m1_y[n3d_index_ex(iz,ix+5, iy)]*c1*ex_Vy0_now[n3d_index_ex(iz,ix+5,iy)]	

	    
									+ ex_m1_x[n3d_index_ex(iz,ix, iy-4)]*c1*ex_Vx0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ ex_m1_x[n3d_index_ex(iz,ix, iy-3)]*c2*ex_Vx0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ ex_m1_x[n3d_index_ex(iz,ix, iy-2)]*c3*ex_Vx0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ ex_m1_x[n3d_index_ex(iz,ix, iy-1)]*c4*ex_Vx0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vx0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_x[n3d_index_ex(iz,ix, iy+1)]*c5*ex_Vx0_now[n3d_index_ex(iz,ix,iy+1)]	
									- ex_m1_x[n3d_index_ex(iz,ix, iy+2)]*c4*ex_Vx0_now[n3d_index_ex(iz,ix,iy+2)]	
									- ex_m1_x[n3d_index_ex(iz,ix, iy+3)]*c3*ex_Vx0_now[n3d_index_ex(iz,ix,iy+3)]	
									- ex_m1_x[n3d_index_ex(iz,ix, iy+4)]*c2*ex_Vx0_now[n3d_index_ex(iz,ix,iy+4)]	
									- ex_m1_x[n3d_index_ex(iz,ix, iy+5)]*c1*ex_Vx0_now[n3d_index_ex(iz,ix,iy+5)]	;		


              ex_sigmaxz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_sigmaxz0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ ex_sigmaxz0_next[n3d_index_ex(iz,ix  , iy)] 
									+ ex_m1_x[n3d_index_ex(iz-4,ix, iy)]*c1*ex_Vx0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ ex_m1_x[n3d_index_ex(iz-3,ix, iy)]*c2*ex_Vx0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ ex_m1_x[n3d_index_ex(iz-2,ix, iy)]*c3*ex_Vx0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ ex_m1_x[n3d_index_ex(iz-1,ix, iy)]*c4*ex_Vx0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vx0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_x[n3d_index_ex(iz+1,ix, iy)]*c5*ex_Vx0_now[n3d_index_ex(iz+1,ix,iy)]	
									- ex_m1_x[n3d_index_ex(iz+2,ix, iy)]*c4*ex_Vx0_now[n3d_index_ex(iz+2,ix,iy)]	
									- ex_m1_x[n3d_index_ex(iz+3,ix, iy)]*c3*ex_Vx0_now[n3d_index_ex(iz+3,ix,iy)]	
									- ex_m1_x[n3d_index_ex(iz+4,ix, iy)]*c2*ex_Vx0_now[n3d_index_ex(iz+4,ix,iy)]	
									- ex_m1_x[n3d_index_ex(iz+5,ix, iy)]*c1*ex_Vx0_now[n3d_index_ex(iz+5,ix,iy)]	
							
									+ ex_m1_z[n3d_index_ex(iz,ix-4, iy)]*c1*ex_Vz0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ ex_m1_z[n3d_index_ex(iz,ix-3, iy)]*c2*ex_Vz0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ ex_m1_z[n3d_index_ex(iz,ix-2, iy)]*c3*ex_Vz0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ ex_m1_z[n3d_index_ex(iz,ix-1, iy)]*c4*ex_Vz0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vz0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_z[n3d_index_ex(iz,ix+1, iy)]*c5*ex_Vz0_now[n3d_index_ex(iz,ix+1,iy)]	
									- ex_m1_z[n3d_index_ex(iz,ix+2, iy)]*c4*ex_Vz0_now[n3d_index_ex(iz,ix+2,iy)]	
									- ex_m1_z[n3d_index_ex(iz,ix+3, iy)]*c3*ex_Vz0_now[n3d_index_ex(iz,ix+3,iy)]	
									- ex_m1_z[n3d_index_ex(iz,ix+4, iy)]*c2*ex_Vz0_now[n3d_index_ex(iz,ix+4,iy)]	
									- ex_m1_z[n3d_index_ex(iz,ix+5, iy)]*c1*ex_Vz0_now[n3d_index_ex(iz,ix+5,iy)]	;						

              ex_sigmayz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*ex_sigmayz0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ ex_sigmayz0_next[n3d_index_ex(iz,ix  , iy)] 
									+ ex_m1_y[n3d_index_ex(iz-4,ix, iy)]*c1*ex_Vy0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ ex_m1_y[n3d_index_ex(iz-3,ix, iy)]*c2*ex_Vy0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ ex_m1_y[n3d_index_ex(iz-2,ix, iy)]*c3*ex_Vy0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ ex_m1_y[n3d_index_ex(iz-1,ix, iy)]*c4*ex_Vy0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vy0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_y[n3d_index_ex(iz+1,ix, iy)]*c5*ex_Vy0_now[n3d_index_ex(iz+1,ix,iy)]	
									- ex_m1_y[n3d_index_ex(iz+2,ix, iy)]*c4*ex_Vy0_now[n3d_index_ex(iz+2,ix,iy)]	
									- ex_m1_y[n3d_index_ex(iz+3,ix, iy)]*c3*ex_Vy0_now[n3d_index_ex(iz+3,ix,iy)]	
									- ex_m1_y[n3d_index_ex(iz+4,ix, iy)]*c2*ex_Vy0_now[n3d_index_ex(iz+4,ix,iy)]	
									- ex_m1_y[n3d_index_ex(iz+5,ix, iy)]*c1*ex_Vy0_now[n3d_index_ex(iz+5,ix,iy)]	
	
									+ ex_m1_z[n3d_index_ex(iz,ix, iy-4)]*c1*ex_Vz0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ ex_m1_z[n3d_index_ex(iz,ix, iy-3)]*c2*ex_Vz0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ ex_m1_z[n3d_index_ex(iz,ix, iy-2)]*c3*ex_Vz0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ ex_m1_z[n3d_index_ex(iz,ix, iy-1)]*c4*ex_Vz0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*ex_Vz0_now[n3d_index_ex(iz,ix,iy)]	
									- ex_m1_z[n3d_index_ex(iz,ix, iy+1)]*c5*ex_Vz0_now[n3d_index_ex(iz,ix,iy+1)]	
									- ex_m1_z[n3d_index_ex(iz,ix, iy+2)]*c4*ex_Vz0_now[n3d_index_ex(iz,ix,iy+2)]	
									- ex_m1_z[n3d_index_ex(iz,ix, iy+3)]*c3*ex_Vz0_now[n3d_index_ex(iz,ix,iy+3)]	
									- ex_m1_z[n3d_index_ex(iz,ix, iy+4)]*c2*ex_Vz0_now[n3d_index_ex(iz,ix,iy+4)]	
									- ex_m1_z[n3d_index_ex(iz,ix, iy+5)]*c1*ex_Vz0_now[n3d_index_ex(iz,ix,iy+5)]	;		

		}
		}

}


int main(int argc, char *argv[]){

	int it, iz, ix, iy;
	int nx = NX;
	int nz = NZ;
	int ny = NY;
	initpar(argc, argv);
	
	//device = 0: using CPU only, 1: using GPU only, 2: using both and check
	int device = atoi(argv[1]);

	//time recording
	float ctime, gtime, ctime_ps, gtime_ps, speedup; //cpu, gpu time; cpu, gpu time per step
	float gpu_kernel_time[3];	// GPU time, [0] for copy in, [1] for computation, [2] for copy out	
	struct timeval start1, end1;
	struct timeval start2, end2;

   	float c1=35.0/294912.0,c2=-405.0/229376.0,c3=567.0/40960.0,c4=-735.0/8192.0,c5=19845.0/16384.0;


	fprintf(stderr, "***********************************************\n");
	fprintf(stderr, "Running 3D Elastic RTM Kernel for %d time steps\n", TS);
	fprintf(stderr, "We record the velocity data every %d time steps\n", Steps_write_back);
	fprintf(stderr, "Data size: (Y = %d , X = %d , Z = %d)\n", ny, nx, nz);
	fprintf(stderr, "***********************************************\n\n\n");

	//********Lin for debug********************************//
	float *debug = NULL;//(float*)malloc(sizeof(float)*(nx)*(nz)*(nt));
	float * tmp = NULL;

	
	//time step +1
	float * ex_Vx0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_Vz0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_Vy0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxx0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmazz0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmayy0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxy0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxz0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmayz0_now = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	

	//Time step +2
	float * ex_Vx0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_Vz0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_Vy0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxx0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmazz0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmayy0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxy0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxz0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmayz0_next = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));


	//output, one step backward, Here records the CPU RTM code, for data checking
	float * ex_Vx0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_Vz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_Vy0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxx0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmazz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmayy0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxy0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmaxz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * ex_sigmayz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));

	//output, one step backward, Here records the output of GPU after certain time step
	float * gpu_ex_Vx0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_Vz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_Vy0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_sigmaxx0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_sigmazz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_sigmayy0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_sigmaxy0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_sigmaxz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float * gpu_ex_sigmayz0_pre = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));

 
   
	//expaned arrays to store different Operators 
	float *ex_m2 = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float *ex_m3 = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float *ex_m2m3 = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float *ex_m1_x = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float *ex_m1_y = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	float *ex_m1_z = (float*)malloc(sizeof(float)*(ny+10)*(nx+10)*(nz+10));



	//set all values to ZERO
	for(iy=0;iy<ny+10;iy++)
	    for(ix=0;ix<nx+10;ix++)
		for(iz=0;iz<nz+10;iz++){	
			ex_Vx0_now[n3d_index_ex_ori(iz,ix,iy)]=0;
			ex_Vy0_now[n3d_index_ex_ori(iz,ix,iy)]=0;
			ex_Vz0_now[n3d_index_ex_ori(iz,ix,iy)]=0;
			ex_sigmaxx0_now[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmazz0_now[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmayy0_now[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxy0_now[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxz0_now[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmayz0_now[n3d_index_ex_ori(iz,ix, iy)]=0;

			ex_Vx0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_Vy0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_Vz0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxx0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmazz0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmayy0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxy0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxz0_next[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmayz0_next[n3d_index_ex_ori(iz,ix, iy)]=0;

			ex_Vx0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_Vy0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_Vz0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxx0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmazz0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmayy0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxy0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmaxz0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_sigmayz0_pre[n3d_index_ex_ori(iz,ix, iy)]=0;


			gpu_ex_Vx0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_Vy0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_Vz0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_sigmaxx0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_sigmayy0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_sigmazz0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_sigmaxy0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_sigmaxz0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			gpu_ex_sigmayz0_pre[n3d_index_ex_ori(iz,ix, iy)] =0;
			

			ex_m2[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_m3[n3d_index_ex_ori(iz,ix, iy)]=0;
			ex_m2m3[n3d_index_ex_ori(iz,ix, iy)]=0; 
			ex_m1_x[n3d_index_ex_ori(iz,ix, iy)]=0; 
			ex_m1_y[n3d_index_ex_ori(iz,ix, iy)]=0; 
			ex_m1_z[n3d_index_ex_ori(iz,ix, iy)]=0; 

	}

	//generate model wavefeild data for constant variables based on model functions from modol-mod.f90
	model_op(ex_m1_y, ex_m1_x, ex_m1_z, ex_m2, ex_m3, ex_m2m3, ny, nx, nz);
	fprintf(stderr, "Model wavefeild initialled\n");

	int nt = TS;
	float *source = (float *)malloc(sizeof(float)*nt);
	source_op2(source, nt);


	//GPU part computation
	if(device ==1 || device == 2){
		fprintf(stderr, "\n\n***********************************************\n");
		fprintf(stderr, "Begin of GPU Computation \n");
//		fprintf(stderr, "Processing:\n");
//		fprintf(stderr, "0%        25%       50%      75%      100%\n");

		rtm_gpu_func(ny, nz, nx, 
		    ex_Vy0_now, ex_Vx0_now, ex_Vz0_now, ex_sigmayy0_now, ex_sigmaxx0_now, ex_sigmazz0_now, ex_sigmaxy0_now, ex_sigmaxz0_now, ex_sigmayz0_now,						//input of time step +1
		    ex_Vy0_next, ex_Vx0_next, ex_Vz0_next, ex_sigmayy0_next, ex_sigmaxx0_next, ex_sigmazz0_next, ex_sigmaxy0_next, ex_sigmaxz0_next, ex_sigmayz0_next,					//input of time step 0
		    gpu_ex_Vy0_pre, gpu_ex_Vx0_pre, gpu_ex_Vz0_pre, gpu_ex_sigmayy0_pre, gpu_ex_sigmaxx0_pre, gpu_ex_sigmazz0_pre, gpu_ex_sigmaxy0_pre, gpu_ex_sigmaxz0_pre, gpu_ex_sigmayz0_pre,	//output of time step -1
		    ex_m1_y, ex_m1_x, ex_m1_z, ex_m2, ex_m3, ex_m2m3,
	            debug, gpu_kernel_time,source);
		fprintf(stderr,"GPU Computing ==============> OK\n");
		fprintf(stderr, "***********************************************\n");

	}


	//CPU part computation 
	if(device==0 || device == 2){
		fprintf(stderr, "\n\n***********************************************\n");
		fprintf(stderr, "CPU Computation Now\n");
		fprintf(stderr, "***********************************************\n");


		gettimeofday(&start1, NULL);

		for(it = 0 ; it<nt; it++){
			gettimeofday(&start2, NULL);
	
			rtm3d_op(ny, nz, nx, 
				    ex_Vy0_now, ex_Vx0_now, ex_Vz0_now, ex_sigmayy0_now, ex_sigmaxx0_now, ex_sigmazz0_now, ex_sigmaxy0_now, ex_sigmaxz0_now, ex_sigmayz0_now,						//input of time step +1
			 	   ex_Vy0_next, ex_Vx0_next, ex_Vz0_next, ex_sigmayy0_next, ex_sigmaxx0_next, ex_sigmazz0_next, ex_sigmaxy0_next, ex_sigmaxz0_next, ex_sigmayz0_next,					//input of time step 0
			 	   ex_Vy0_pre, ex_Vx0_pre, ex_Vz0_pre, ex_sigmayy0_pre, ex_sigmaxx0_pre, ex_sigmazz0_pre, ex_sigmaxy0_pre, ex_sigmaxz0_pre, ex_sigmayz0_pre,	//output of time step -1
			 	   ex_m1_y, ex_m1_x, ex_m1_z, ex_m2, ex_m3, ex_m2m3);
	
			gettimeofday(&end2, NULL);
	
			ctime = (end2.tv_sec-start2.tv_sec)+(end2.tv_usec-start2.tv_usec)/1000000.0;
			fprintf(stderr, "CPU time at %d / %d step:  %.8f\n",it+1, nt,  ctime);

			if(it+1==Steps_write_back){
				gettimeofday(&end1, NULL);
			}
			

#ifdef CPU_FILE
			//Write CPU data to History file
			if(it%Steps_write_back==0 ){
			fprintf(stderr,"CPU Computation Write to History File...\n");
	
			int ierr = srite("data_Vx0",ex_Vx0_pre, sizeof(float)*(nx+10)*(ny+10)*(nz+10));
			ierr = srite("data_Vy0",ex_Vy0_pre, sizeof(float)*(nx+10)*(ny+10)*(nz+10));
			ierr = srite("data_Vz0",ex_Vz0_pre, sizeof(float)*(nx+10)*(ny+10)*(nz+10));
			}
#endif

			if(it < nt-1){

				//Add Source
				add_source(source[it], 75, 100, 100, ny, nx, nz, ex_sigmayy0_now);
				add_source(source[it], 75, 100, 100, ny, nx, nz, ex_sigmaxx0_now);
				add_source(source[it], 75, 100, 100, ny, nx, nz, ex_sigmazz0_now);
		 	
				//Change Pointer
				tmp = ex_Vx0_pre; ex_Vx0_pre = ex_Vx0_now; ex_Vx0_now = tmp;
				tmp = ex_Vx0_pre; ex_Vx0_pre = ex_Vx0_next; ex_Vx0_next = tmp; 
				
				tmp = ex_Vz0_pre; ex_Vz0_pre = ex_Vz0_now; ex_Vz0_now = tmp;
				tmp = ex_Vz0_pre; ex_Vz0_pre = ex_Vz0_next; ex_Vz0_next = tmp; 
		
				tmp = ex_Vy0_pre; ex_Vy0_pre = ex_Vy0_now; ex_Vy0_now = tmp;
				tmp = ex_Vy0_pre; ex_Vy0_pre = ex_Vy0_next; ex_Vy0_next = tmp; 
		
				tmp = ex_sigmaxx0_pre; ex_sigmaxx0_pre = ex_sigmaxx0_now; ex_sigmaxx0_now = tmp;
				tmp = ex_sigmaxx0_pre; ex_sigmaxx0_pre = ex_sigmaxx0_next; ex_sigmaxx0_next = tmp; 
			
				tmp = ex_sigmazz0_pre; ex_sigmazz0_pre = ex_sigmazz0_now; ex_sigmazz0_now = tmp;
				tmp = ex_sigmazz0_pre; ex_sigmazz0_pre = ex_sigmazz0_next; ex_sigmazz0_next = tmp; 
			
				tmp = ex_sigmayy0_pre; ex_sigmayy0_pre = ex_sigmayy0_now; ex_sigmayy0_now = tmp;
				tmp = ex_sigmayy0_pre; ex_sigmayy0_pre = ex_sigmayy0_next; ex_sigmayy0_next = tmp; 
			
				tmp = ex_sigmaxy0_pre; ex_sigmaxy0_pre = ex_sigmaxy0_now; ex_sigmaxy0_now = tmp;
				tmp = ex_sigmaxy0_pre; ex_sigmaxy0_pre = ex_sigmaxy0_next; ex_sigmaxy0_next = tmp; 
		
				tmp = ex_sigmaxz0_pre; ex_sigmaxz0_pre = ex_sigmaxz0_now; ex_sigmaxz0_now = tmp;
				tmp = ex_sigmaxz0_pre; ex_sigmaxz0_pre = ex_sigmaxz0_next; ex_sigmaxz0_next = tmp; 
			
				tmp = ex_sigmayz0_pre; ex_sigmayz0_pre = ex_sigmayz0_now; ex_sigmayz0_now = tmp;
				tmp = ex_sigmayz0_pre; ex_sigmayz0_pre = ex_sigmayz0_next; ex_sigmayz0_next = tmp; 
		
			}

		}
	
	fprintf(stderr,"CPU Computing ==============> OK\n");
	}


	if(device==2){
	
		//Reuslt check by comparing GPU output with CPu output
		func_check(ny, nx, nz, ex_sigmaxx0_pre, gpu_ex_sigmaxx0_pre);
		func_check(ny, nx, nz, ex_sigmayy0_pre, gpu_ex_sigmayy0_pre);
		func_check(ny, nx, nz, ex_sigmazz0_pre, gpu_ex_sigmazz0_pre);
		func_check(ny, nx, nz, ex_Vx0_pre, gpu_ex_Vx0_pre);
		func_check(ny, nx, nz, ex_Vy0_pre, gpu_ex_Vy0_pre);
		func_check(ny, nx, nz, ex_Vz0_pre, gpu_ex_Vz0_pre);
		func_check(ny, nx, nz, ex_sigmaxy0_pre, gpu_ex_sigmaxy0_pre);
		func_check(ny, nx, nz, ex_sigmaxz0_pre, gpu_ex_sigmaxz0_pre);
		func_check(ny, nx, nz, ex_sigmayz0_pre, gpu_ex_sigmayz0_pre);
	
		//test write to file
		//FILE *fp = fopen("Vx0","ab+");
		fprintf(stderr, "<<<<<<<<<<<<<<<<<PERFORMANCE PROFILING>>>>>>>>>>>>>>>>\n");
		fprintf(stderr, "Using %d GPU cards and %d CPU threads\n", GPU_NUM, omp_get_num_threads());
	
		ctime = (end1.tv_sec-start1.tv_sec)+(end1.tv_usec-start1.tv_usec)/1000000.0;
		fprintf(stderr, "For every %d time steps:\n",   Steps_write_back);
		fprintf(stderr, "CPU computing time:\t  %.8f\n", ctime);
		fprintf(stderr, "GPU computing time:\t  %.8f\n", gpu_kernel_time[1]);
		fprintf(stderr, "CPU data to GPU:   \t  %.8f\n", gpu_kernel_time[0]);
		fprintf(stderr, "GPU data to CPU:   \t  %.8f\n", gpu_kernel_time[2]);
		
		
		fprintf(stderr, "Computing   Speedup: %.2f\n",(ctime/gpu_kernel_time[1])); 
		fprintf(stderr, "Application Speedup: %.2f\n",(ctime/(gpu_kernel_time[1]+gpu_kernel_time[0]+gpu_kernel_time[2]))); 
		
		fprintf(stderr, "<<<<<<<<<<<<<<<<<PERFORMANCE PROFILING>>>>>>>>>>>>>>>>\n");
	}

	free(ex_Vx0_now);
	free(ex_Vz0_now);
	free(ex_Vy0_now);
	free(ex_sigmaxx0_now);
	free(ex_sigmazz0_now);
	free(ex_sigmayy0_now);
	free(ex_sigmaxy0_now);
	free(ex_sigmaxz0_now);
	free(ex_sigmayz0_now);

	free(ex_Vx0_next);
	free(ex_Vz0_next);
	free(ex_Vy0_next);
	free(ex_sigmaxx0_next);
	free(ex_sigmazz0_next);
	free(ex_sigmayy0_next);
	free(ex_sigmaxy0_next);
	free(ex_sigmaxz0_next);
	free(ex_sigmayz0_next);


	free(ex_Vx0_pre);
	free(ex_Vz0_pre);
	free(ex_Vy0_pre);
	free(ex_sigmaxx0_pre);
	free(ex_sigmazz0_pre);
	free(ex_sigmayy0_pre);
	free(ex_sigmaxy0_pre);
	free(ex_sigmaxz0_pre);
	free(ex_sigmayz0_pre);


	free(ex_m2);
	free(ex_m3);
	free(ex_m2m3);
	free(ex_m1_x);
	free(ex_m1_z);
	free(ex_m1_y);

	return NULL;
}


