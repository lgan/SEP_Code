#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu.h"
#include "sep3d.h"
#include "seplib.h"




__constant__ float g_source[TS];

__global__ void rtm_gpu_kernel(int ny, int nz, int nx, int gpu_id, int it, 
        float *g_ex_Vy0_now,  float * g_ex_Vx0_now, float * g_ex_Vz0_now, float * g_ex_sigmayy0_now, float *g_ex_sigmaxx0_now, float * g_ex_sigmazz0_now, float * g_ex_sigmaxy0_now, float * g_ex_sigmaxz0_now, float * g_ex_sigmayz0_now,//(nz, nx, nt)
        float *g_ex_Vy0_next,  float * g_ex_Vx0_next, float * g_ex_Vz0_next, float * g_ex_sigmayy0_next, float *g_ex_sigmaxx0_next, float * g_ex_sigmazz0_next, float * g_ex_sigmaxy0_next, float * g_ex_sigmaxz0_next, float * g_ex_sigmayz0_next,//(nz, nx, nt)
        float *g_ex_Vy0_pre,  float * g_ex_Vx0_pre, float * g_ex_Vz0_pre, float * g_ex_sigmayy0_pre, float *g_ex_sigmaxx0_pre, float * g_ex_sigmazz0_pre, float * g_ex_sigmaxy0_pre, float * g_ex_sigmaxz0_pre, float * g_ex_sigmayz0_pre,//(nz, nx, nt)
     	float * g_ex_m1_y,    float * g_ex_m1_x,    float * g_ex_m1_z,   float *  g_ex_m2,  float * g_ex_m3,  float * g_ex_m2m3);//(nz+10,	nx+10)




extern "C" void setup_cuda(int n_gpus){
	int dr;
	int i, j, k;

	for(i=0; i<n_gpus; i++) device[i] = i;

	for(i=0; i<n_gpus; i++) {
		cudaDeviceSynchronize();
		
		cudaSetDevice(device[i]);
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device[i]);
		cudaDriverGetVersion(&dr);

		//Enable P2P memcopies between GPUs
		if(n_gpus >1){
			for(j=0;j<n_gpus; j++){
				if(i == j) continue;
				int peer_access_available = 0;
				cudaDeviceCanAccessPeer(&peer_access_available, device[i], device[j]);
				if(peer_access_available)
					cudaDeviceEnablePeerAccess(device[j], 0);
			}
		}
	}

}


__global__ void add_source_kernel(int ny, int nz, int nx, int source_iy, int source_iz, int source_ix, int it, float * g_ex_sigmayy0_now, float *g_ex_sigmaxx0_now, float * g_ex_sigmazz0_now)//(ny+10, nx+10, nz+10)
{
	float dt = 0.4*(4.0)/(sqrt(2.0)*2000.0);

	int iz, ix, iy;
	iz = blockIdx.x*blockDim.x + threadIdx.x;
	ix = blockIdx.y*blockDim.y + threadIdx.y;
	iy = blockIdx.z*blockDim.z + threadIdx.z;

	//add_source
	if(iy==source_iy && ix==source_ix && iz==source_iz){
		g_ex_sigmayy0_now[n3d_index_ex(iz,ix  ,iy)] += dt*g_source[it];
		g_ex_sigmaxx0_now[n3d_index_ex(iz,ix  ,iy)] += dt*g_source[it];
		g_ex_sigmazz0_now[n3d_index_ex(iz,ix  ,iy)] += dt*g_source[it];
	}

}


__global__ void rtm_gpu_kernel(int ny, int nz, int nx,//int source_iy, int source_ix, int source_iz,
        float *g_ex_Vy0_now,  float * g_ex_Vx0_now, float * g_ex_Vz0_now, float * g_ex_sigmayy0_now, float *g_ex_sigmaxx0_now, float * g_ex_sigmazz0_now, float * g_ex_sigmaxy0_now, float * g_ex_sigmaxz0_now, float * g_ex_sigmayz0_now,//(ny+10, nx+10, nz+10)
        float *g_ex_Vy0_next,  float * g_ex_Vx0_next, float * g_ex_Vz0_next, float * g_ex_sigmayy0_next, float *g_ex_sigmaxx0_next, float * g_ex_sigmazz0_next, float * g_ex_sigmaxy0_next, float * g_ex_sigmaxz0_next, float * g_ex_sigmayz0_next,//(ny+10,nx+10,nz+10)
        float *g_ex_Vy0_pre,  float * g_ex_Vx0_pre, float * g_ex_Vz0_pre, float * g_ex_sigmayy0_pre, float *g_ex_sigmaxx0_pre, float * g_ex_sigmazz0_pre, float * g_ex_sigmaxy0_pre, float * g_ex_sigmaxz0_pre, float * g_ex_sigmayz0_pre,//(ny+10,nx+10,nz+10) 
     	float * g_ex_m1_y,    float * g_ex_m1_x,    float * g_ex_m1_z,  float * g_ex_m2, float * g_ex_m3,  float * g_ex_m2m3)//(ny+10,nx+10,nz+10) 
{

	float c1=35.0/294912.0,c2=-405.0/229376.0,c3=567.0/40960.0,c4=-735.0/8192.0,c5=19845.0/16384.0;

	//GPU thread index
	int iz, ix, iy;
	iz = blockIdx.x*blockDim.x + threadIdx.x;
	ix = blockIdx.y*blockDim.y + threadIdx.y;
	iy = blockIdx.z*blockDim.z + threadIdx.z;

//	__shared__ float sh_g_ex_Vx0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_Vy0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_Vz0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_sigmaxx0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_sigmayy0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_sigmazz0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_sigmaxy0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_sigmayz0_now[(TX+10)*(TZ+10)*(TY+10)];
//	__shared__ float sh_g_ex_sigmaxz0_now[(TX+10)*(TZ+10)*(TY+10)];
//
//
//	sh_g_ex_Vx0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_Vx0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_Vy0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_Vy0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_Vz0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_Vz0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_sigmaxx0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_sigmaxx0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_sigmayy0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_sigmayy0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_sigmazz0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_sigmazz0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_sigmaxy0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_sigmaxy0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_sigmaxz0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_sigmaxz0_pre[n3d_index_ex(iz, ix, iy)];
//	sh_g_ex_sigmayz0_now[n3d_index_blk_ex(threadIdx.x, threadIdx.y, threadIdx.z)] = g_ex_sigmayz0_pre[n3d_index_ex(iz, ix, iy)];
 

       	g_ex_Vx0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_Vx0_pre[n3d_index_ex(iz,ix  ,iy)]*/	+ g_ex_Vx0_next[n3d_index_ex(iz, ix, iy)]	

									+ g_ex_m2m3[n3d_index_ex(iz,ix-5, iy)]*c1*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix-5,iy)]							
							 		+ g_ex_m2m3[n3d_index_ex(iz,ix-4, iy)]*c2*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m2m3[n3d_index_ex(iz,ix-3, iy)]*c3*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix-2, iy)]*c4*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix-1, iy)]*c5*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix-1,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+1, iy)]*c4*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+2, iy)]*c3*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+3, iy)]*c2*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+4, iy)]*c1*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix+4,iy)]
	

									+ g_ex_m2[n3d_index_ex(iz,ix-5, iy)]*c1*g_ex_sigmayy0_now[n3d_index_ex(iz,ix-5,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix-4, iy)]*c2*g_ex_sigmayy0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m2[n3d_index_ex(iz,ix-3, iy)]*c3*g_ex_sigmayy0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-2, iy)]*c4*g_ex_sigmayy0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-1, iy)]*c5*g_ex_sigmayy0_now[n3d_index_ex(iz,ix-1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,  ix, iy)]*c5*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+1, iy)]*c4*g_ex_sigmayy0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+2, iy)]*c3*g_ex_sigmayy0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+3, iy)]*c2*g_ex_sigmayy0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+4, iy)]*c1*g_ex_sigmayy0_now[n3d_index_ex(iz,ix+4,iy)]	
	

									+ g_ex_m2[n3d_index_ex(iz,ix-5, iy)]*c1*g_ex_sigmazz0_now[n3d_index_ex(iz,ix-5,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix-4, iy)]*c2*g_ex_sigmazz0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m2[n3d_index_ex(iz,ix-3, iy)]*c3*g_ex_sigmazz0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-2, iy)]*c4*g_ex_sigmazz0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-1, iy)]*c5*g_ex_sigmazz0_now[n3d_index_ex(iz,ix-1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,  ix, iy)]*c5*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+1, iy)]*c4*g_ex_sigmazz0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+2, iy)]*c3*g_ex_sigmazz0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+3, iy)]*c2*g_ex_sigmazz0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+4, iy)]*c1*g_ex_sigmazz0_now[n3d_index_ex(iz,ix+4,iy)]	
	

									+ g_ex_m3[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy+5)]							
	

									+ g_ex_m3[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_sigmaxz0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m3[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_sigmaxz0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_sigmaxz0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_sigmaxz0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,  ix, iy)]*c5*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_sigmaxz0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_sigmaxz0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_sigmaxz0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_sigmaxz0_now[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_sigmaxz0_now[n3d_index_ex(iz+5,ix,iy)]	;						
	


         	g_ex_Vy0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_Vy0_pre[n3d_index_ex(iz,ix  ,iy)]*/	+ g_ex_Vy0_next[n3d_index_ex(iz, ix, iy)]	

									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-5)]*c1*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-5)]							
							 		+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-4)]*c2*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-3)]*c3*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-2)]*c4*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-1)]*c5*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy-1)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+1)]*c4*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+2)]*c3*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+3)]*c2*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+4)]*c1*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy+4)]
	

									+ g_ex_m2[n3d_index_ex(iz,ix, iy-5)]*c1*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-5)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix, iy-4)]*c2*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-3)]*c3*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-2)]*c4*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-1)]*c5*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy-1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+1)]*c4*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+2)]*c3*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+3)]*c2*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+4)]*c1*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy+4)]	
	

									+ g_ex_m2[n3d_index_ex(iz,ix, iy-5)]*c1*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-5)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix, iy-4)]*c2*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-3)]*c3*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-2)]*c4*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-1)]*c5*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy-1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+1)]*c4*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+2)]*c3*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+3)]*c2*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+4)]*c1*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy+4)]	
	

									+ g_ex_m3[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_sigmayz0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m3[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_sigmayz0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_sigmayz0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_sigmayz0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_sigmayz0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_sigmayz0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_sigmayz0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_sigmayz0_now[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_sigmayz0_now[n3d_index_ex(iz+5,ix,iy)]							
	

									+ g_ex_m3[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m3[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_sigmaxy0_now[n3d_index_ex(iz,ix+5,iy)]	;						




         	g_ex_Vz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_Vz0_pre[n3d_index_ex(iz,ix  ,iy)]*/	+ g_ex_Vz0_next[n3d_index_ex(iz, ix, iy)]	

									+ g_ex_m2m3[n3d_index_ex(iz-5,ix, iy)]*c1*g_ex_sigmazz0_now[n3d_index_ex(iz-5,ix,iy)]							
							 		+ g_ex_m2m3[n3d_index_ex(iz-4,ix, iy)]*c2*g_ex_sigmazz0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m2m3[n3d_index_ex(iz-3,ix, iy)]*c3*g_ex_sigmazz0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz-2,ix, iy)]*c4*g_ex_sigmazz0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz-1,ix, iy)]*c5*g_ex_sigmazz0_now[n3d_index_ex(iz-1,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmazz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+1,ix, iy)]*c4*g_ex_sigmazz0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+2,ix, iy)]*c3*g_ex_sigmazz0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+3,ix, iy)]*c2*g_ex_sigmazz0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+4,ix, iy)]*c1*g_ex_sigmazz0_now[n3d_index_ex(iz+4,ix,iy)]
	

									+ g_ex_m2[n3d_index_ex(iz-5,ix, iy)]*c1*g_ex_sigmaxx0_now[n3d_index_ex(iz-5,ix,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz-4,ix, iy)]*c2*g_ex_sigmaxx0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m2[n3d_index_ex(iz-3,ix, iy)]*c3*g_ex_sigmaxx0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-2,ix, iy)]*c4*g_ex_sigmaxx0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-1,ix, iy)]*c5*g_ex_sigmaxx0_now[n3d_index_ex(iz-1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxx0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+1,ix, iy)]*c4*g_ex_sigmaxx0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+2,ix, iy)]*c3*g_ex_sigmaxx0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+3,ix, iy)]*c2*g_ex_sigmaxx0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+4,ix, iy)]*c1*g_ex_sigmaxx0_now[n3d_index_ex(iz+4,ix,iy)]
	

									+ g_ex_m2[n3d_index_ex(iz-5,ix, iy)]*c1*g_ex_sigmayy0_now[n3d_index_ex(iz-5,ix,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz-4,ix, iy)]*c2*g_ex_sigmayy0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m2[n3d_index_ex(iz-3,ix, iy)]*c3*g_ex_sigmayy0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-2,ix, iy)]*c4*g_ex_sigmayy0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-1,ix, iy)]*c5*g_ex_sigmayy0_now[n3d_index_ex(iz-1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+1,ix, iy)]*c4*g_ex_sigmayy0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+2,ix, iy)]*c3*g_ex_sigmayy0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+3,ix, iy)]*c2*g_ex_sigmayy0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+4,ix, iy)]*c1*g_ex_sigmayy0_now[n3d_index_ex(iz+4,ix,iy)]
	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_sigmayz0_now[n3d_index_ex(iz,ix,iy+5)]							
	

									+ g_ex_m3[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m3[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_sigmaxz0_now[n3d_index_ex(iz,ix+5,iy)]	;						


		

              g_ex_sigmaxx0_pre[n3d_index_ex(iz,ix  ,iy)] =/* g_ex_sigmaxx0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmaxx0_next[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_x[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_Vx0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m1_x[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_Vx0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_Vx0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_Vx0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_Vx0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_Vx0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_Vx0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_Vx0_now[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_Vx0_now[n3d_index_ex(iz,ix+5,iy)]	;						

	    
              g_ex_sigmayy0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmayy0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmayy0_next[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy+5)]	;		


              g_ex_sigmazz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmazz0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmazz0_next[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_z[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_Vz0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m1_z[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_Vz0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_Vz0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_Vz0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_Vz0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_Vz0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_Vz0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_Vz0_now[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_Vz0_now[n3d_index_ex(iz+5,ix,iy)]	;		 
	
	


              g_ex_sigmaxy0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmaxy0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmaxy0_next[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_y[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_Vy0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m1_y[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_Vy0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_Vy0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_Vy0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_Vy0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_Vy0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_Vy0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_Vy0_now[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_Vy0_now[n3d_index_ex(iz,ix+5,iy)]	

	    
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy+5)]	;		


              g_ex_sigmaxz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmaxz0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmaxz0_next[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_x[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_Vx0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m1_x[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_Vx0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_Vx0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_Vx0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vx0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_Vx0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_Vx0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_Vx0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_Vx0_now[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_Vx0_now[n3d_index_ex(iz+5,ix,iy)]	
							
									+ g_ex_m1_z[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_Vz0_now[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m1_z[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_Vz0_now[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_Vz0_now[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_Vz0_now[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_Vz0_now[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_Vz0_now[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_Vz0_now[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_Vz0_now[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_Vz0_now[n3d_index_ex(iz,ix+5,iy)]	;						


              g_ex_sigmayz0_pre[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmayz0_pre[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmayz0_next[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_y[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_Vy0_now[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m1_y[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_Vy0_now[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_Vy0_now[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_Vy0_now[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vy0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_Vy0_now[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_Vy0_now[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_Vy0_now[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_Vy0_now[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_Vy0_now[n3d_index_ex(iz+5,ix,iy)]	
	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_Vz0_now[n3d_index_ex(iz,ix,iy+5)]	;		

}


extern "C" void rtm_gpu_func(int ny, int nz, int nx, 
        float *ex_Vy0_now,  float * ex_Vx0_now, float * ex_Vz0_now, float * ex_sigmayy0_now, float *ex_sigmaxx0_now, float * ex_sigmazz0_now, float * ex_sigmaxy0_now, float * ex_sigmaxz0_now, float * ex_sigmayz0_now,//(ny+10, nx+10, nx+10)
        float *ex_Vy0_next,  float * ex_Vx0_next, float * ex_Vz0_next, float * ex_sigmayy0_next, float *ex_sigmaxx0_next, float * ex_sigmazz0_next, float * ex_sigmaxy0_next, float * ex_sigmaxz0_next, float * ex_sigmayz0_next,//(ny+10, nx+10, nx+10) 
        float *ex_Vy0_pre,  float * ex_Vx0_pre, float * ex_Vz0_pre, float * ex_sigmayy0_pre, float *ex_sigmaxx0_pre, float * ex_sigmazz0_pre, float * ex_sigmaxy0_pre, float * ex_sigmaxz0_pre, float * ex_sigmayz0_pre,//(ny+10, nx+10, nx+10) 
        float * ex_m1_y, float * ex_m1_x,float * ex_m1_z,float * ex_m2, float * ex_m3, float * ex_m2m3,//)//(nz+10,nx+10)
	float * debug, float * gpu_kernel_time, float *source)

{	
     	cudaError_t err;
	cudaEvent_t start1, start2, start3, stop1, stop2, stop3;
	float elapsedTime1 = 0.0f;//time for data copy in GPU
	float elapsedTime2 = 0.0f;//time for kernel computing + overlapping
	float elapsedTime3 = 0.0f;//time for data copy out GPU

	cudaSetDevice(0);
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);
	cudaEventCreate(&stop3);
	
	//config multiple GPU status
	//Acquire the number of GPU available
	int n_gpus;
#ifdef GPU_NUM
	n_gpus = GPU_NUM;
#else
	fprintf(stderr, "Please enter the number of GPU to use (between 1 and %d):\n", MAX_NUM_GPUS);
	scanf("%d", &n_gpus);
#endif

	int gpu_n;
	cudaGetDeviceCount(&gpu_n);
	if(n_gpus>gpu_n){
		fprintf(stderr, "[EXIT] GPU number avalable %d less than number assigned %d\n", gpu_n, n_gpus);
		//fprintf(stderr, "Set GPU number from assigned %d to GPU number avalable %d \n", n_gpus, gpu_n);
		//n_gpus = gpu_n;
		exit(1);
	}

	//init GPU devices with P2P being enabled
	setup_cuda(n_gpus);

	// GPU inputs and outputs corresponds to CPU inputs and outputs
	float *g_ex_Vx0_now[n_gpus], *g_ex_Vx0_next[n_gpus], *g_ex_Vx0_pre[n_gpus];
	float *g_ex_Vy0_now[n_gpus], *g_ex_Vy0_next[n_gpus], *g_ex_Vy0_pre[n_gpus];
	float *g_ex_Vz0_now[n_gpus], *g_ex_Vz0_next[n_gpus], *g_ex_Vz0_pre[n_gpus];
	float *g_ex_sigmaxx0_now[n_gpus], *g_ex_sigmaxx0_next[n_gpus], *g_ex_sigmaxx0_pre[n_gpus];
	float *g_ex_sigmayy0_now[n_gpus], *g_ex_sigmayy0_next[n_gpus], *g_ex_sigmayy0_pre[n_gpus];
	float *g_ex_sigmazz0_now[n_gpus], *g_ex_sigmazz0_next[n_gpus], *g_ex_sigmazz0_pre[n_gpus];
	float *g_ex_sigmaxy0_now[n_gpus], *g_ex_sigmaxy0_next[n_gpus], *g_ex_sigmaxy0_pre[n_gpus];
	float *g_ex_sigmaxz0_now[n_gpus], *g_ex_sigmaxz0_next[n_gpus], *g_ex_sigmaxz0_pre[n_gpus];
	float *g_ex_sigmayz0_now[n_gpus], *g_ex_sigmayz0_next[n_gpus], *g_ex_sigmayz0_pre[n_gpus];

	float *g_ex_m2[n_gpus];
	float *g_ex_m3[n_gpus];
	float *g_ex_m2m3[n_gpus];
	float *g_ex_m1_x[n_gpus];
	float *g_ex_m1_z[n_gpus];
	float *g_ex_m1_y[n_gpus];
	//config index and offset

	int n1 = (NZ+10), n2 = (NX+10), n3 = (NY+10); // NZ -->NX--> NY: fast --> slow, NX, NY, NZ refers to the original index from Fortran code, and here needs to be extended by plus 10 for stencil doundary
	n3 = (n3 - 2*radius)/n_gpus + 2*radius;		
	int n_bytes_gpu = (n1*n2*n3)*sizeof(float);   //bytes length assigned for each GPU

	int start[n_gpus];				//CPU index for copy data to GPU	
	for(int i=0; i<n_gpus; i++){			//Define coordinates and offsets parameters	
		start[i] = i*(n3-2*radius) * (n1*n2);
	}



	//determin which GPU to add source (75, 100, 100) based on the number of gpus
	int gpu_id=0;
	int source_y=75;
	int n3_source = ((n3-2*radius)); 
	while((n3_source)*(gpu_id+1)<75){
		gpu_id++;
		source_y -= n3_source;
	}
	printf("%d, %d, %d\n", gpu_id, n3_source, source_y);


	//offset, bytes length, and block/grid config for internal part
	int offset_internal = radius*(n1*n2);		
	int n_bytes_gpu_internal = (n3-4*radius)*(n1*n2)*sizeof(float);
	dim3 dimGrid_internal(n1/TZ, n2/TX, (n3-4*radius)/TY);
	dim3 dimBlock(TZ, TX, TY);

	//offset, bytes length, and block/grid config for upper and bottom halo parts
	int offset_halo_up = 0;	
	int offset_halo_bt = (n3-3*radius)*(n1*n2);
	int n_bytes_gpu_halo = radius*(n1*n2)*sizeof(float);
	dim3 dimGrid_halo(n1/TZ, n2/TX, radius/TY);

	//offset and bytes length for output data back to CPU
	int offset_out = radius * n1*n2;
	int n_bytes_gpu_back = n1*n2*(n3-2*radius)*sizeof(float);

	//malloc data on each GPU
	for(int i=0; i<n_gpus; i++){
		cudaSetDevice(device[i]);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaMalloc((void **) &g_ex_Vx0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vx0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vx0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_Vy0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vy0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vy0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_Vz0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vz0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vz0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmaxx0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxx0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxx0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmayy0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayy0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayy0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmazz0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmazz0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmazz0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmaxy0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxy0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxy0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmaxz0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxz0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxz0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmayz0_now[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayz0_next[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayz0_pre[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_m2[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m3[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m2m3[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m1_x[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m1_y[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m1_z[i], n_bytes_gpu);
		
	}
	
	//copy data to each GPU
	cudaSetDevice(0);
	cudaEventRecord(start1, 0);
	for(int i = 0; i<n_gpus; i++){

		cudaSetDevice(device[i]);

		//put source on the second GPU (point with global index (75, 100, 100) )
		if(device[i]==gpu_id){
			cudaMemcpyToSymbol(g_source,source,TS*sizeof(float));
		}

		cudaMemcpy(g_ex_Vx0_now[i],  &ex_Vx0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vx0_next[i], &ex_Vx0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vx0_pre[i], &ex_Vx0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_Vy0_now[i],  &ex_Vy0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vy0_next[i], &ex_Vy0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vy0_pre[i], &ex_Vy0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_Vz0_now[i],  &ex_Vz0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vz0_next[i], &ex_Vz0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vz0_pre[i], &ex_Vz0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmaxx0_now[i], &ex_sigmaxx0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxx0_next[i], &ex_sigmaxx0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxx0_pre[i], &ex_sigmaxx0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmayy0_now[i], &ex_sigmayy0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayy0_next[i], &ex_sigmayy0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayy0_pre[i], &ex_sigmayy0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmazz0_now[i], &ex_sigmazz0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmazz0_next[i], &ex_sigmazz0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmazz0_pre[i], &ex_sigmazz0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmaxy0_now[i], &ex_sigmaxy0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxy0_next[i], &ex_sigmaxy0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxy0_pre[i], &ex_sigmaxy0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmaxz0_now[i], &ex_sigmaxz0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxz0_next[i], &ex_sigmaxz0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxz0_pre[i], &ex_sigmaxz0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmayz0_now[i], &ex_sigmayz0_now[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayz0_next[i], &ex_sigmayz0_next[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayz0_pre[i], &ex_sigmayz0_pre[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_m2[i], &ex_m2[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m3[i], &ex_m3[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m2m3[i], &ex_m2m3[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m1_x[i], &ex_m1_x[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m1_y[i], &ex_m1_y[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m1_z[i], &ex_m1_z[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
	}
	cudaSetDevice(0);
	cudaEventRecord(stop1, 0);
	

	//Define separate streams for overlapping communication
	cudaStream_t stream_halo[n_gpus], stream_internal[n_gpus];
	for(int i=0; i<n_gpus; i++){
		cudaSetDevice(device[i]);
		cudaStreamCreate(&stream_halo[i]);
		cudaStreamCreate(&stream_internal[i]);
	}

//-----------------------------------------------------------------------------//
//--------------------------GPU COMPUTATION------------------------------------//
//-----------------------------------------------------------------------------//

	//fprintf(stderr,"GPU Computing ... ...(NZ=%d, NX=%d, NY=%d, TZ=%d, TX=%d, TY=%d)\n", nz, nx, ny, TZ, TX, TY);
	
	cudaSetDevice(0);
	cudaEventRecord(start2, 0);

//Doing elastic RTM for TS time steps
	//fprintf(stderr, "[");
	for(int g_it=0; g_it<TS;g_it++){
		fprintf(stderr, "GPU at %d / %d step\n",g_it+1, TS);
	
		//Calculate the halo regions first
		for(int i=0; i<n_gpus; i++){
			cudaSetDevice(i);

			//Upper Halo
			rtm_gpu_kernel<<<dimGrid_halo, dimBlock,0,stream_halo[i]>>>(ny, nz, nx,
				g_ex_Vy0_now[i] + offset_halo_up, g_ex_Vx0_now[i] + offset_halo_up, g_ex_Vz0_now[i] + offset_halo_up, g_ex_sigmayy0_now[i] + offset_halo_up, g_ex_sigmaxx0_now[i] + offset_halo_up, g_ex_sigmazz0_now[i] + offset_halo_up, g_ex_sigmaxy0_now[i] + offset_halo_up, g_ex_sigmaxz0_now[i] + offset_halo_up, g_ex_sigmayz0_now[i] + offset_halo_up,
				g_ex_Vy0_next[i] + offset_halo_up, g_ex_Vx0_next[i] + offset_halo_up, g_ex_Vz0_next[i] + offset_halo_up, g_ex_sigmayy0_next[i] + offset_halo_up, g_ex_sigmaxx0_next[i] + offset_halo_up, g_ex_sigmazz0_next[i] + offset_halo_up, g_ex_sigmaxy0_next[i] + offset_halo_up, g_ex_sigmaxz0_next[i] + offset_halo_up, g_ex_sigmayz0_next[i] + offset_halo_up,
				g_ex_Vy0_pre[i] + offset_halo_up, g_ex_Vx0_pre[i] + offset_halo_up, g_ex_Vz0_pre[i] + offset_halo_up, g_ex_sigmayy0_pre[i] + offset_halo_up, g_ex_sigmaxx0_pre[i] + offset_halo_up, g_ex_sigmazz0_pre[i] + offset_halo_up, g_ex_sigmaxy0_pre[i] + offset_halo_up, g_ex_sigmaxz0_pre[i] + offset_halo_up, g_ex_sigmayz0_pre[i] + offset_halo_up,
				g_ex_m1_y[i] + offset_halo_up, g_ex_m1_x[i] + offset_halo_up, g_ex_m1_z[i] + offset_halo_up, g_ex_m2[i] + offset_halo_up, g_ex_m3[i] + offset_halo_up, g_ex_m2m3[i] + offset_halo_up);

			//Botom Halo
			rtm_gpu_kernel<<<dimGrid_halo, dimBlock,0,stream_halo[i]>>>(ny, nz, nx,
				g_ex_Vy0_now[i] + offset_halo_bt, g_ex_Vx0_now[i] + offset_halo_bt, g_ex_Vz0_now[i] + offset_halo_bt, g_ex_sigmayy0_now[i] + offset_halo_bt, g_ex_sigmaxx0_now[i] + offset_halo_bt, g_ex_sigmazz0_now[i] + offset_halo_bt, g_ex_sigmaxy0_now[i] + offset_halo_bt, g_ex_sigmaxz0_now[i] + offset_halo_bt, g_ex_sigmayz0_now[i] + offset_halo_bt,
				g_ex_Vy0_next[i] + offset_halo_bt, g_ex_Vx0_next[i] + offset_halo_bt, g_ex_Vz0_next[i] + offset_halo_bt, g_ex_sigmayy0_next[i] + offset_halo_bt, g_ex_sigmaxx0_next[i] + offset_halo_bt, g_ex_sigmazz0_next[i] + offset_halo_bt, g_ex_sigmaxy0_next[i] + offset_halo_bt, g_ex_sigmaxz0_next[i] + offset_halo_bt, g_ex_sigmayz0_next[i] + offset_halo_bt,
				g_ex_Vy0_pre[i] + offset_halo_bt, g_ex_Vx0_pre[i] + offset_halo_bt, g_ex_Vz0_pre[i] + offset_halo_bt, g_ex_sigmayy0_pre[i] + offset_halo_bt, g_ex_sigmaxx0_pre[i] + offset_halo_bt, g_ex_sigmazz0_pre[i] + offset_halo_bt, g_ex_sigmaxy0_pre[i] + offset_halo_bt, g_ex_sigmaxz0_pre[i] + offset_halo_bt, g_ex_sigmayz0_pre[i] + offset_halo_bt,
				g_ex_m1_y[i] + offset_halo_bt, g_ex_m1_x[i] + offset_halo_bt, g_ex_m1_z[i] + offset_halo_bt, g_ex_m2[i] + offset_halo_bt, g_ex_m3[i] + offset_halo_bt, g_ex_m2m3[i] + offset_halo_bt);

			cudaStreamQuery(stream_halo[i]);
		}	
	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error5: %s.\n", cudaGetErrorString(err));
		exit(0);
	}	
	

		//Compute the internal part
		for(int i=0; i<n_gpus; i++){
			cudaSetDevice(i);	

			rtm_gpu_kernel<<<dimGrid_internal, dimBlock, 0, stream_internal[i]>>>(ny, nz, nx,
				g_ex_Vy0_now[i] + offset_internal, g_ex_Vx0_now[i] + offset_internal, g_ex_Vz0_now[i] + offset_internal, g_ex_sigmayy0_now[i] + offset_internal, g_ex_sigmaxx0_now[i] + offset_internal, g_ex_sigmazz0_now[i] + offset_internal, g_ex_sigmaxy0_now[i] + offset_internal, g_ex_sigmaxz0_now[i] + offset_internal, g_ex_sigmayz0_now[i] + offset_internal,
				g_ex_Vy0_next[i] + offset_internal, g_ex_Vx0_next[i] + offset_internal, g_ex_Vz0_next[i] + offset_internal, g_ex_sigmayy0_next[i] + offset_internal, g_ex_sigmaxx0_next[i] + offset_internal, g_ex_sigmazz0_next[i] + offset_internal, g_ex_sigmaxy0_next[i] + offset_internal, g_ex_sigmaxz0_next[i] + offset_internal, g_ex_sigmayz0_next[i] + offset_internal,
			g_ex_Vy0_pre[i] + offset_internal, g_ex_Vx0_pre[i] + offset_internal, g_ex_Vz0_pre[i] + offset_internal, g_ex_sigmayy0_pre[i] + offset_internal, g_ex_sigmaxx0_pre[i] + offset_internal, g_ex_sigmazz0_pre[i] + offset_internal, g_ex_sigmaxy0_pre[i] + offset_internal, g_ex_sigmaxz0_pre[i] + offset_internal, g_ex_sigmayz0_pre[i] + offset_internal,
			g_ex_m1_y[i] + offset_internal, g_ex_m1_x[i] + offset_internal, g_ex_m1_z[i] + offset_internal, g_ex_m2[i] + offset_internal, g_ex_m3[i] + offset_internal, g_ex_m2m3[i] + offset_internal);
		}	

		//Add source to point at GPU i based on the number of GPUs
	
	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error5: %s.\n", cudaGetErrorString(err));
		exit(0);
	}	

		cudaSetDevice(gpu_id);
		dim3 dimGrid_source(n1/TZ, n2/TX, (n3-2*radius)/TY);
		add_source_kernel<<<dimGrid_source, dimBlock>>>(ny, nz, nx, source_y, 100, 100, g_it, g_ex_sigmayy0_now[gpu_id], g_ex_sigmaxx0_now[gpu_id], g_ex_sigmazz0_now[gpu_id]);
	
	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error5: %s.\n", cudaGetErrorString(err));
		exit(0);
	}	

		//Halo updating
		if(g_it < TS-1){

			//offset for each halo parts
			int offset_down_snd = (n3-2*radius)*(n1*n2);
			int offset_down_rcv = 0;//n3-2*radius)*(n1*n2);
			int offset_up_snd   = radius*n1*n2;
			int offset_up_rcv   = (n3-radius)*n1*n2;
	
			//Send halos downwards
			for(int i=0; i<n_gpus-1; i++){
				cudaMemcpyPeerAsync(g_ex_Vx0_pre[i+1]+offset_down_rcv,i+1, g_ex_Vx0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_Vy0_pre[i+1]+offset_down_rcv,i+1, g_ex_Vy0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_Vz0_pre[i+1]+offset_down_rcv,i+1, g_ex_Vz0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmaxx0_pre[i+1]+offset_down_rcv,i+1, g_ex_sigmaxx0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmayy0_pre[i+1]+offset_down_rcv,i+1, g_ex_sigmayy0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmazz0_pre[i+1]+offset_down_rcv,i+1, g_ex_sigmazz0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmaxy0_pre[i+1]+offset_down_rcv,i+1, g_ex_sigmaxy0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmaxz0_pre[i+1]+offset_down_rcv,i+1, g_ex_sigmaxz0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmayz0_pre[i+1]+offset_down_rcv,i+1, g_ex_sigmayz0_pre[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
			}		
			
			//Synchronize to avoid stalling
			for(int i=0; i<n_gpus; i++){
				cudaSetDevice(i);
				cudaStreamSynchronize(stream_halo[i]);
			}	

			//Send halos upwards
			for(int i=1; i<n_gpus; i++){
				cudaMemcpyPeerAsync(g_ex_Vx0_pre[i-1]+offset_up_rcv,i-1, g_ex_Vx0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_Vy0_pre[i-1]+offset_up_rcv,i-1, g_ex_Vy0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_Vz0_pre[i-1]+offset_up_rcv,i-1, g_ex_Vz0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmaxx0_pre[i-1]+offset_up_rcv,i-1, g_ex_sigmaxx0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmayy0_pre[i-1]+offset_up_rcv,i-1, g_ex_sigmayy0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmazz0_pre[i-1]+offset_up_rcv,i-1, g_ex_sigmazz0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmaxz0_pre[i-1]+offset_up_rcv,i-1, g_ex_sigmaxz0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmayz0_pre[i-1]+offset_up_rcv,i-1, g_ex_sigmayz0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
				cudaMemcpyPeerAsync(g_ex_sigmaxy0_pre[i-1]+offset_up_rcv,i-1, g_ex_sigmaxy0_pre[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
			}
		}
	
		//Here we only record the kernel computing time between 0 ~ (Steps_write_back-1)
		if(g_it+1 == Steps_write_back){ 
			cudaSetDevice(0);
			cudaEventRecord(stop2, 0);
		}


		//Write data back to CPU and write to History Files evert Steps_write_back steps
		if((g_it+1)%Steps_write_back==0 ){	
			//Copy out data back to CPU
		
			if((g_it+1)==Steps_write_back){
				cudaSetDevice(0);
				cudaEventRecord(start3, 0);
			}
	
			for(int i = 0; i<n_gpus; i++){
		
				cudaSetDevice(device[i]);
				cudaMemcpy(&ex_Vx0_pre[start[i]]+offset_out, g_ex_Vx0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_Vy0_pre[start[i]]+offset_out, g_ex_Vy0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_Vz0_pre[start[i]]+offset_out, g_ex_Vz0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_sigmaxx0_pre[start[i]]+offset_out, g_ex_sigmaxx0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_sigmayy0_pre[start[i]]+offset_out, g_ex_sigmayy0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_sigmazz0_pre[start[i]]+offset_out, g_ex_sigmazz0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_sigmaxy0_pre[start[i]]+offset_out, g_ex_sigmaxy0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_sigmayz0_pre[start[i]]+offset_out, g_ex_sigmayz0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
				cudaMemcpy(&ex_sigmaxz0_pre[start[i]]+offset_out, g_ex_sigmaxz0_pre[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
			}


			if((g_it+1)==Steps_write_back){
				cudaSetDevice(0);
				cudaEventRecord(stop3, 0);
			}
	
#ifdef GPU_FILE		
			fprintf(stderr,"GPU to CPU and then write to History File at step %d\n", g_it+1);

			int ierr = srite("g_data_Vx0",ex_Vx0_pre, sizeof(float)*(nx+10)*(ny+10)*(nz+10));
			ierr = srite("g_data_Vy0",ex_Vy0_pre, sizeof(float)*(nx+10)*(ny+10)*(nz+10));
			ierr = srite("g_data_Vz0",ex_Vz0_pre, sizeof(float)*(nx+10)*(ny+10)*(nz+10));
#endif

		}	

		//Change pointer
		if(g_it < TS -1){
			float *g_tmp = NULL;
			for(int i=0; i<n_gpus; i++){
				cudaSetDevice(device[i]);
	
				g_tmp = g_ex_Vx0_pre[i]; g_ex_Vx0_pre[i] = g_ex_Vx0_now[i]; g_ex_Vx0_now[i] = g_tmp;
				g_tmp = g_ex_Vx0_pre[i]; g_ex_Vx0_pre[i] = g_ex_Vx0_next[i]; g_ex_Vx0_next[i] = g_tmp; 
	
				g_tmp = g_ex_Vy0_pre[i]; g_ex_Vy0_pre[i] = g_ex_Vy0_now[i]; g_ex_Vy0_now[i] = g_tmp;
				g_tmp = g_ex_Vy0_pre[i]; g_ex_Vy0_pre[i] = g_ex_Vy0_next[i]; g_ex_Vy0_next[i] = g_tmp; 
	
				g_tmp = g_ex_Vz0_pre[i]; g_ex_Vz0_pre[i] = g_ex_Vz0_now[i]; g_ex_Vz0_now[i] = g_tmp;
				g_tmp = g_ex_Vz0_pre[i]; g_ex_Vz0_pre[i] = g_ex_Vz0_next[i]; g_ex_Vz0_next[i] = g_tmp;
	 
				g_tmp = g_ex_sigmaxx0_pre[i]; g_ex_sigmaxx0_pre[i] = g_ex_sigmaxx0_now[i]; g_ex_sigmaxx0_now[i] = g_tmp;
				g_tmp = g_ex_sigmaxx0_pre[i]; g_ex_sigmaxx0_pre[i] = g_ex_sigmaxx0_next[i]; g_ex_sigmaxx0_next[i] = g_tmp;
	 
				g_tmp = g_ex_sigmayy0_pre[i]; g_ex_sigmayy0_pre[i] = g_ex_sigmayy0_now[i]; g_ex_sigmayy0_now[i] = g_tmp;
				g_tmp = g_ex_sigmayy0_pre[i]; g_ex_sigmayy0_pre[i] = g_ex_sigmayy0_next[i]; g_ex_sigmayy0_next[i] = g_tmp;
	 
				g_tmp = g_ex_sigmazz0_pre[i]; g_ex_sigmazz0_pre[i] = g_ex_sigmazz0_now[i]; g_ex_sigmazz0_now[i] = g_tmp;
				g_tmp = g_ex_sigmazz0_pre[i]; g_ex_sigmazz0_pre[i] = g_ex_sigmazz0_next[i]; g_ex_sigmazz0_next[i] = g_tmp;
	 
				g_tmp = g_ex_sigmaxy0_pre[i]; g_ex_sigmaxy0_pre[i] = g_ex_sigmaxy0_now[i]; g_ex_sigmaxy0_now[i] = g_tmp;
				g_tmp = g_ex_sigmaxy0_pre[i]; g_ex_sigmaxy0_pre[i] = g_ex_sigmaxy0_next[i]; g_ex_sigmaxy0_next[i] = g_tmp; 
	
				g_tmp = g_ex_sigmaxz0_pre[i]; g_ex_sigmaxz0_pre[i] = g_ex_sigmaxz0_now[i]; g_ex_sigmaxz0_now[i] = g_tmp;
				g_tmp = g_ex_sigmaxz0_pre[i]; g_ex_sigmaxz0_pre[i] = g_ex_sigmaxz0_next[i]; g_ex_sigmaxz0_next[i] = g_tmp;
	 
				g_tmp = g_ex_sigmayz0_pre[i]; g_ex_sigmayz0_pre[i] = g_ex_sigmayz0_now[i]; g_ex_sigmayz0_now[i] = g_tmp;
				g_tmp = g_ex_sigmayz0_pre[i]; g_ex_sigmayz0_pre[i] = g_ex_sigmayz0_next[i]; g_ex_sigmayz0_next[i] = g_tmp; 
	
			}
		}
		
		//if(g_it%(TS/40)==0) fprintf(stderr, "#");
	}
	//fprintf(stderr, "]\n");	

	cudaEventSynchronize(stop1);
	cudaEventSynchronize(stop2);
	cudaEventSynchronize(stop3);

	cudaEventElapsedTime(&elapsedTime1, start1, stop1);
	cudaEventElapsedTime(&elapsedTime2, start2, stop2);
	cudaEventElapsedTime(&elapsedTime3, start3, stop3);


	gpu_kernel_time[0] = (float)(elapsedTime1/1000.);
	gpu_kernel_time[1] = (float)(elapsedTime2/1000.);
	gpu_kernel_time[2] = (float)(elapsedTime3/1000.);
	
	cudaEventDestroy(start1);
	cudaEventDestroy(start2);
	cudaEventDestroy(start3);
	cudaEventDestroy(stop1);
	cudaEventDestroy(stop2);
	cudaEventDestroy(stop3);
	
	for(int i=0; i<n_gpus; i++){
		cudaSetDevice(i);

		cudaFree(g_ex_Vx0_now[i]);
		cudaFree(g_ex_Vz0_now[i]);
		cudaFree(g_ex_Vy0_now[i]);
		cudaFree(g_ex_sigmaxx0_now[i]);
		cudaFree(g_ex_sigmazz0_now[i]);
		cudaFree(g_ex_sigmayy0_now[i]);
		cudaFree(g_ex_sigmaxy0_now[i]);
		cudaFree(g_ex_sigmaxz0_now[i]);
		cudaFree(g_ex_sigmayz0_now[i]);
		
		//Time step +2
		cudaFree(g_ex_Vx0_next[i]);
		cudaFree(g_ex_Vz0_next[i]);
		cudaFree(g_ex_Vy0_next[i]);
		cudaFree(g_ex_sigmaxx0_next[i]);
		cudaFree(g_ex_sigmazz0_next[i]);
		cudaFree(g_ex_sigmayy0_next[i]);
		cudaFree(g_ex_sigmaxy0_next[i]);
		cudaFree(g_ex_sigmaxz0_next[i]);
		cudaFree(g_ex_sigmayz0_next[i]);
	
	
		//time step 0 and output
		cudaFree(g_ex_Vx0_pre[i]);
		cudaFree(g_ex_Vz0_pre[i]);
		cudaFree(g_ex_Vy0_pre[i]);
		cudaFree(g_ex_sigmaxx0_pre[i]);
		cudaFree(g_ex_sigmazz0_pre[i]);
		cudaFree(g_ex_sigmayy0_pre[i]);
		cudaFree(g_ex_sigmaxy0_pre[i]);
		cudaFree(g_ex_sigmaxz0_pre[i]);
		cudaFree(g_ex_sigmayz0_pre[i]);
	   
		//expaned arrays to store different Operators 
		cudaFree(g_ex_m2[i]);
		cudaFree(g_ex_m3[i]);
		cudaFree(g_ex_m2m3[i]);
		cudaFree(g_ex_m1_x[i]);
		cudaFree(g_ex_m1_y[i]);
		cudaFree(g_ex_m1_z[i]);
		}
}

