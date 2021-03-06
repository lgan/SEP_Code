#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu.h"

__global__ void rtm_gpu_kernel(int ny, int nz, int nx,
        float *g_ex_Vy0_in,  float * g_ex_Vx0_in, float * g_ex_Vz0_in, float * g_ex_sigmayy0_in, float *g_ex_sigmaxx0_in, float * g_ex_sigmazz0_in, float * g_ex_sigmaxy0_in, float * g_ex_sigmaxz0_in, float * g_ex_sigmayz0_in,//(nz, nx, nt)
        float *g_ex_Vy0_in1,  float * g_ex_Vx0_in1, float * g_ex_Vz0_in1, float * g_ex_sigmayy0_in1, float *g_ex_sigmaxx0_in1, float * g_ex_sigmazz0_in1, float * g_ex_sigmaxy0_in1, float * g_ex_sigmaxz0_in1, float * g_ex_sigmayz0_in1,//(nz, nx, nt)
        float *g_ex_Vy0_out,  float * g_ex_Vx0_out, float * g_ex_Vz0_out, float * g_ex_sigmayy0_out, float *g_ex_sigmaxx0_out, float * g_ex_sigmazz0_out, float * g_ex_sigmaxy0_out, float * g_ex_sigmaxz0_out, float * g_ex_sigmayz0_out,//(nz, nx, nt)
     	float * g_ex_m1_y,    float * g_ex_m1_x,    float * g_ex_m1_z,   float *  g_ex_m2,  float * g_ex_m3,  float * g_ex_m2m3);//(nz+10,	nx+10)




extern "C" void setup_cuda(int n_gpus){
	
fprintf(stderr, "Today, we are using %d GPUs; specifically: \n", n_gpus);
	int dr;
	int i, j, k;

	for(i=0; i<n_gpus; i++) device[i] = i;

	for(i=0; i<n_gpus; i++) {
		cudaDeviceSynchronize();
		
		cudaSetDevice(device[i]);
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device[i]);
		cudaDriverGetVersion(&dr);

		fprintf(stderr, "   GPU %s (%d),", properties.name, device[i]);
		if(properties.unifiedAddressing) fprintf(stderr, " UVA initiated, ");
		else fprintf(stderr, " UVA not working,");
		fprintf(stderr, "driver %d\n", dr);

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





extern "C" void rtm_kernel_multi(int ny, int nz, int nx, int n_gpus) 
{
}

#ifndef MULTI_GPUS
extern "C" void rtm_gpu_init(int ny, int nz, int nx, int n_gpus) 
{	
	cudaError_t cuda_ret;
     	cudaError_t err;

	//Set Device 
    	cuda_ret = cudaSetDevice(1);
	if(cuda_ret != cudaSuccess){
		fprintf(stderr, "Failed to Set The cuda Device !\n");
		exit(0);
	}
	else{
		fprintf(stderr, "GPU Device Set ====> OK\n");
	}

	// data init
	
	//Time step +1
	cudaMalloc(&g_ex_Vx0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_Vz0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_Vy0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxx0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmazz0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmayy0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxy0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxz0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmayz0_in, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	
	//Time step +2
	cudaMalloc(&g_ex_Vx0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_Vz0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_Vy0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxx0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmazz0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmayy0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxy0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxz0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmayz0_in1, sizeof(float)*(ny+10)*(nx+10)*(nz+10));


	//time step 0 and output
	cudaMalloc(&g_ex_Vx0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_Vz0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_Vy0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxx0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmazz0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmayy0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxy0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmaxz0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_sigmayz0_out, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
   
	//expaned arrays to store different Operators 
	cudaMalloc(&g_ex_m2, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_m3, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_m2m3, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_m1_x, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_m1_y, sizeof(float)*(ny+10)*(nx+10)*(nz+10));
	cudaMalloc(&g_ex_m1_z, sizeof(float)*(ny+10)*(nx+10)*(nz+10));

	cudaFuncSetCacheConfig(rtm_gpu_kernel,cudaFuncCachePreferShared);
	
	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error0: %s.\n", cudaGetErrorString(err));
		exit(0);
	}else{	
		fprintf(stderr,"GPU Data Init ====> OK\n");
	}
	// data copy

}



extern "C" void rtm_gpu_copy_in(int ny, int nz, int nx, 
        float *ex_Vy0_in,  float * ex_Vx0_in, float * ex_Vz0_in, float * ex_sigmayy0_in, float *ex_sigmaxx0_in, float * ex_sigmazz0_in, float * ex_sigmaxy0_in, float * ex_sigmaxz0_in, float * ex_sigmayz0_in,//(nz, nx, nt)
        float *ex_Vy0_in1,  float * ex_Vx0_in1, float * ex_Vz0_in1, float * ex_sigmayy0_in1, float *ex_sigmaxx0_in1, float * ex_sigmazz0_in1, float * ex_sigmaxy0_in1, float * ex_sigmaxz0_in1, float * ex_sigmayz0_in1,//(nz, nx, nt)
        float *ex_Vy0_out,  float * ex_Vx0_out, float * ex_Vz0_out, float * ex_sigmayy0_out, float *ex_sigmaxx0_out, float * ex_sigmazz0_out, float * ex_sigmaxy0_out, float * ex_sigmaxz0_out, float * ex_sigmayz0_out,//(nz, nx, nt)
     	float * ex_m1_y,  float * ex_m1_x, float * ex_m1_z, float * ex_m2, float * ex_m3, float * ex_m2m3)//(nz+10,	nx+10)
  {	
     	cudaError_t err;
	
	// data copy

	cudaMemcpy(g_ex_Vy0_in, ex_Vy0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_Vx0_in, ex_Vx0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_Vz0_in, ex_Vz0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxx0_in, ex_sigmaxx0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmayy0_in, ex_sigmayy0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxy0_in, ex_sigmaxy0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmayz0_in, ex_sigmayz0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxz0_in, ex_sigmaxz0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmazz0_in, ex_sigmazz0_in, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);

	cudaMemcpy(g_ex_Vy0_in1, ex_Vy0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_Vx0_in1, ex_Vx0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_Vz0_in1, ex_Vz0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxx0_in1, ex_sigmaxx0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmayy0_in1, ex_sigmayy0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxy0_in1, ex_sigmaxy0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmayz0_in1, ex_sigmayz0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxz0_in1, ex_sigmaxz0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmazz0_in1, ex_sigmazz0_in1, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);

	cudaMemcpy(g_ex_Vy0_out, ex_Vy0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_Vx0_out, ex_Vx0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_Vz0_out, ex_Vz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxx0_out, ex_sigmaxx0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmayy0_out, ex_sigmayy0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxy0_out, ex_sigmaxy0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmayz0_out, ex_sigmayz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmaxz0_out, ex_sigmaxz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_sigmazz0_out, ex_sigmazz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), cudaMemcpyHostToDevice);



	cudaMemcpy(g_ex_m1_y, ex_m1_y, sizeof(float)*(ny+10)*(nx+10)*(nz+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_m1_x, ex_m1_x, sizeof(float)*(ny+10)*(nx+10)*(nz+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_m1_z, ex_m1_z, sizeof(float)*(ny+10)*(nx+10)*(nz+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_m2, ex_m2, sizeof(float)*(ny+10)*(nx+10)*(nz+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_m3, ex_m3, sizeof(float)*(ny+10)*(nx+10)*(nz+10), cudaMemcpyHostToDevice);
	cudaMemcpy(g_ex_m2m3, ex_m2m3, sizeof(float)*(ny+10)*(nx+10)*(nz+10), cudaMemcpyHostToDevice);
	
	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error1: %s.\n", cudaGetErrorString(err));
		exit(0);
		
	}else{
		fprintf(stderr,"Data Copy To GPU  ====> OK\n");
	}
}



extern "C" void rtm_gpu_copy_out(int ny, int nz, int nx, 
        float *ex_Vy0_out,  float * ex_Vx0_out, float * ex_Vz0_out, float * ex_sigmayy0_out, float *ex_sigmaxx0_out, float * ex_sigmazz0_out, float * ex_sigmaxy0_out, float * ex_sigmaxz0_out, float * ex_sigmayz0_out)//(nz, nx, nt)
{	
     	cudaError_t err;
	
	// data copy back from GPU mem
	cudaMemcpy(ex_Vy0_out, g_ex_Vy0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10),  		cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_Vx0_out, g_ex_Vx0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10),  		cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_Vz0_out, g_ex_Vz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), 			cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_sigmaxx0_out, g_ex_sigmaxx0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), 	cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_sigmayy0_out, g_ex_sigmayy0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), 	cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_sigmaxy0_out, g_ex_sigmaxy0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), 	cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_sigmaxz0_out, g_ex_sigmaxz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), 	cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_sigmayz0_out, g_ex_sigmayz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), 	cudaMemcpyDeviceToHost);
	cudaMemcpy(ex_sigmazz0_out, g_ex_sigmazz0_out, sizeof(float)*(nx+10)*(nz+10)*(ny+10), 	cudaMemcpyDeviceToHost);
	//cudaMemcpy(sigmazz0, g_sigmazz0,  sizeof(float)*nx*nz*nt, 	cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error2: %s.\n", cudaGetErrorString(err));
		exit(0);
	}else{
	fprintf(stderr,"Data Copy To CPU ====> OK\n");
	}
}


extern "C" void rtm_gpu_final()
{

	//release GPU memory space

     	cudaError_t err;
	
	cudaFree(g_ex_Vx0_in);
	cudaFree(g_ex_Vz0_in);
	cudaFree(g_ex_Vy0_in);
	cudaFree(g_ex_sigmaxx0_in);
	cudaFree(g_ex_sigmazz0_in);
	cudaFree(g_ex_sigmayy0_in);
	cudaFree(g_ex_sigmaxy0_in);
	cudaFree(g_ex_sigmaxz0_in);
	cudaFree(g_ex_sigmayz0_in);
	
	//Time step +2
	cudaFree(g_ex_Vx0_in1);
	cudaFree(g_ex_Vz0_in1);
	cudaFree(g_ex_Vy0_in1);
	cudaFree(g_ex_sigmaxx0_in1);
	cudaFree(g_ex_sigmazz0_in1);
	cudaFree(g_ex_sigmayy0_in1);
	cudaFree(g_ex_sigmaxy0_in1);
	cudaFree(g_ex_sigmaxz0_in1);
	cudaFree(g_ex_sigmayz0_in1);


	//time step 0 and output
	cudaFree(g_ex_Vx0_out);
	cudaFree(g_ex_Vz0_out);
	cudaFree(g_ex_Vy0_out);
	cudaFree(g_ex_sigmaxx0_out);
	cudaFree(g_ex_sigmazz0_out);
	cudaFree(g_ex_sigmayy0_out);
	cudaFree(g_ex_sigmaxy0_out);
	cudaFree(g_ex_sigmaxz0_out);
	cudaFree(g_ex_sigmayz0_out);
   
	//expaned arrays to store different Operators 
	cudaFree(g_ex_m2);
	cudaFree(g_ex_m3);
	cudaFree(g_ex_m2m3);
	cudaFree(g_ex_m1_x);
	cudaFree(g_ex_m1_y);
	cudaFree(g_ex_m1_z);

	err = cudaGetLastError();

	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error3: %s.\n", cudaGetErrorString(err));
		exit(0);
	}else{
	fprintf(stderr,"GPU Mem Released ====> OK\n");
	}
}


void rtm_gpu_change_pointer(int n_gpus){
		
			
		fprintf(stderr, "GPU pointer changed\n");

		__device__ float *g_tmp = NULL;

		
		for(int i=0; i<n_gpus; i++){
		cudaSetDevice[i];
		g_tmp = 	g_ex_Vx0_out[i];
		g_ex_Vx0_out[i] = 	g_ex_Vx0_in[i];
		g_ex_Vx0_in[i] = 	g_tmp;

		g_tmp = 	g_ex_Vx0_out[i];
		g_ex_Vx0_out[i] = 	g_ex_Vx0_in1[i];
		g_ex_Vx0_in1[i] = 	g_tmp; 

	

//		g_tmp = 	g_ex_Vz0_out;
//		g_ex_Vz0_out = 	g_ex_Vz0_in;
//		g_ex_Vz0_in = 	g_tmp;
//
//		g_tmp = 	g_ex_Vz0_out;
//		g_ex_Vz0_out = 	g_ex_Vz0_in1;
//		g_ex_Vz0_in1 = 	g_tmp; 
//
//
//	
//		g_tmp = 	g_ex_Vy0_out;
//		g_ex_Vy0_out = 	g_ex_Vy0_in;
//		g_ex_Vy0_in = 	g_tmp;
//
//		g_tmp = 	g_ex_Vy0_out;
//		g_ex_Vy0_out = 	g_ex_Vy0_in1;
//		g_ex_Vy0_in1 = 	g_tmp; 

	

		g_tmp = 	g_ex_sigmaxx0_out[i];
		g_ex_sigmaxx0_out[i] = g_ex_sigmaxx0_in[i];
		g_ex_sigmaxx0_in[i] = g_tmp;

		g_tmp = g_ex_sigmaxx0_out[i];
		g_ex_sigmaxx0_out[i] = g_ex_sigmaxx0_in1[i];
		g_ex_sigmaxx0_in1[i] = g_tmp; 

	
		g_tmp = g_ex_sigmazz0_out[i];
		g_ex_sigmazz0_out[i] = g_ex_sigmazz0_in[i];
		g_ex_sigmazz0_in[i] = g_tmp;

		g_tmp = g_ex_sigmazz0_out[i];
		g_ex_sigmazz0_out[i] = g_ex_sigmazz0_in1[i];
		g_ex_sigmazz0_in1[i] = g_tmp; 


	
		g_tmp = g_ex_sigmayy0_out[i];
		g_ex_sigmayy0_out[i] = g_ex_sigmayy0_in[i];
		g_ex_sigmayy0_in[i] = g_tmp;

		g_tmp = g_ex_sigmayy0_out[i];
		g_ex_sigmayy0_out[i] = g_ex_sigmayy0_in1[i];
		g_ex_sigmayy0_in1[i] = g_tmp; 


	
		g_tmp = g_ex_sigmaxy0_out[i];
		g_ex_sigmaxy0_out[i] = g_ex_sigmaxy0_in[i];
		g_ex_sigmaxy0_in[i] = g_tmp;

		g_tmp = g_ex_sigmaxy0_out[i];
		g_ex_sigmaxy0_out[i] = g_ex_sigmaxy0_in1[i];
		g_ex_sigmaxy0_in1[i] = g_tmp; 



	
		g_tmp = g_ex_sigmaxz0_out[i];
		g_ex_sigmaxz0_out[i] = g_ex_sigmaxz0_in[i];
		g_ex_sigmaxz0_in[i] = g_tmp;

		g_tmp = g_ex_sigmaxz0_out[i];
		g_ex_sigmaxz0_out[i] = g_ex_sigmaxz0_in1[i];
		g_ex_sigmaxz0_in1[i] = g_tmp; 


	
		g_tmp = g_ex_sigmayz0_out[i];
		g_ex_sigmayz0_out[i] = g_ex_sigmayz0_in[i];
		g_ex_sigmayz0_in[i] = g_tmp;
		
		g_tmp = g_ex_sigmayz0_out[i];
		g_ex_sigmayz0_out[i] = g_ex_sigmayz0_in1[i];
		g_ex_sigmayz0_in1[i] = g_tmp; 
		}

}

#endif

__global__ void rtm_gpu_kernel(int ny, int nz, int nx,
        float *g_ex_Vy0_in,  float * g_ex_Vx0_in, float * g_ex_Vz0_in, float * g_ex_sigmayy0_in, float *g_ex_sigmaxx0_in, float * g_ex_sigmazz0_in, float * g_ex_sigmaxy0_in, float * g_ex_sigmaxz0_in, float * g_ex_sigmayz0_in,//(nz, nx, nt)
        float *g_ex_Vy0_in1,  float * g_ex_Vx0_in1, float * g_ex_Vz0_in1, float * g_ex_sigmayy0_in1, float *g_ex_sigmaxx0_in1, float * g_ex_sigmazz0_in1, float * g_ex_sigmaxy0_in1, float * g_ex_sigmaxz0_in1, float * g_ex_sigmayz0_in1,//(nz, nx, nt)
        float *g_ex_Vy0_out,  float * g_ex_Vx0_out, float * g_ex_Vz0_out, float * g_ex_sigmayy0_out, float *g_ex_sigmaxx0_out, float * g_ex_sigmazz0_out, float * g_ex_sigmaxy0_out, float * g_ex_sigmaxz0_out, float * g_ex_sigmayz0_out,//(nz, nx, nt)
     	float * g_ex_m1_y,    float * g_ex_m1_x,    float * g_ex_m1_z,  float * g_ex_m2, float * g_ex_m3,  float * g_ex_m2m3)//(nz+10,	nx+10)
{

	float c1=35.0/294912.0,c2=-405.0/229376.0,c3=567.0/40960.0,c4=-735.0/8192.0,c5=19845.0/16384.0;

	//GPU thread index
	int iz, ix, iy;
	iz = blockIdx.x*blockDim.x + threadIdx.x;
	ix = blockIdx.y*blockDim.y + threadIdx.y;
	iy = blockIdx.z*blockDim.z + threadIdx.z;
	//gt = it;
        	g_ex_Vx0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_Vx0_out[n3d_index_ex(iz,ix  ,iy)]*/	+ g_ex_Vx0_in1[n3d_index_ex(iz, ix, iy)]	

									+ g_ex_m2m3[n3d_index_ex(iz,ix-5, iy)]*c1*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix-5,iy)]							
							 		+ g_ex_m2m3[n3d_index_ex(iz,ix-4, iy)]*c2*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m2m3[n3d_index_ex(iz,ix-3, iy)]*c3*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix-2, iy)]*c4*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix-1, iy)]*c5*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix-1,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+1, iy)]*c4*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+2, iy)]*c3*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+3, iy)]*c2*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix+4, iy)]*c1*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix+4,iy)]
	

									+ g_ex_m2[n3d_index_ex(iz,ix-5, iy)]*c1*g_ex_sigmayy0_in[n3d_index_ex(iz,ix-5,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix-4, iy)]*c2*g_ex_sigmayy0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m2[n3d_index_ex(iz,ix-3, iy)]*c3*g_ex_sigmayy0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-2, iy)]*c4*g_ex_sigmayy0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-1, iy)]*c5*g_ex_sigmayy0_in[n3d_index_ex(iz,ix-1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,  ix, iy)]*c5*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+1, iy)]*c4*g_ex_sigmayy0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+2, iy)]*c3*g_ex_sigmayy0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+3, iy)]*c2*g_ex_sigmayy0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+4, iy)]*c1*g_ex_sigmayy0_in[n3d_index_ex(iz,ix+4,iy)]	
	

									+ g_ex_m2[n3d_index_ex(iz,ix-5, iy)]*c1*g_ex_sigmazz0_in[n3d_index_ex(iz,ix-5,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix-4, iy)]*c2*g_ex_sigmazz0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m2[n3d_index_ex(iz,ix-3, iy)]*c3*g_ex_sigmazz0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-2, iy)]*c4*g_ex_sigmazz0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m2[n3d_index_ex(iz,ix-1, iy)]*c5*g_ex_sigmazz0_in[n3d_index_ex(iz,ix-1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,  ix, iy)]*c5*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+1, iy)]*c4*g_ex_sigmazz0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+2, iy)]*c3*g_ex_sigmazz0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+3, iy)]*c2*g_ex_sigmazz0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix+4, iy)]*c1*g_ex_sigmazz0_in[n3d_index_ex(iz,ix+4,iy)]	
	

									+ g_ex_m3[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy+5)]							
	

									+ g_ex_m3[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_sigmaxz0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m3[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_sigmaxz0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_sigmaxz0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_sigmaxz0_in[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,  ix, iy)]*c5*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_sigmaxz0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_sigmaxz0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_sigmaxz0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_sigmaxz0_in[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_sigmaxz0_in[n3d_index_ex(iz+5,ix,iy)]	;						
	


         	g_ex_Vy0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_Vy0_out[n3d_index_ex(iz,ix  ,iy)]*/	+ g_ex_Vy0_in1[n3d_index_ex(iz, ix, iy)]	

									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-5)]*c1*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy-5)]							
							 		+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-4)]*c2*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-3)]*c3*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-2)]*c4*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m2m3[n3d_index_ex(iz,ix, iy-1)]*c5*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy-1)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+1)]*c4*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+2)]*c3*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+3)]*c2*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy+4)]*c1*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy+4)]
	

									+ g_ex_m2[n3d_index_ex(iz,ix, iy-5)]*c1*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy-5)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix, iy-4)]*c2*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-3)]*c3*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-2)]*c4*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-1)]*c5*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy-1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+1)]*c4*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+2)]*c3*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+3)]*c2*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+4)]*c1*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy+4)]	
	

									+ g_ex_m2[n3d_index_ex(iz,ix, iy-5)]*c1*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy-5)]							
							 		+ g_ex_m2[n3d_index_ex(iz,ix, iy-4)]*c2*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-3)]*c3*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-2)]*c4*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m2[n3d_index_ex(iz,ix, iy-1)]*c5*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy-1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+1)]*c4*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+2)]*c3*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+3)]*c2*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy+4)]*c1*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy+4)]	
	

									+ g_ex_m3[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_sigmayz0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m3[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_sigmayz0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_sigmayz0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_sigmayz0_in[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_sigmayz0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_sigmayz0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_sigmayz0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_sigmayz0_in[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_sigmayz0_in[n3d_index_ex(iz+5,ix,iy)]							
	

									+ g_ex_m3[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m3[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_sigmaxy0_in[n3d_index_ex(iz,ix+5,iy)]	;						




         	g_ex_Vz0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_Vz0_out[n3d_index_ex(iz,ix  ,iy)]*/	+ g_ex_Vz0_in1[n3d_index_ex(iz, ix, iy)]	

									+ g_ex_m2m3[n3d_index_ex(iz-5,ix, iy)]*c1*g_ex_sigmazz0_in[n3d_index_ex(iz-5,ix,iy)]							
							 		+ g_ex_m2m3[n3d_index_ex(iz-4,ix, iy)]*c2*g_ex_sigmazz0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m2m3[n3d_index_ex(iz-3,ix, iy)]*c3*g_ex_sigmazz0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz-2,ix, iy)]*c4*g_ex_sigmazz0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m2m3[n3d_index_ex(iz-1,ix, iy)]*c5*g_ex_sigmazz0_in[n3d_index_ex(iz-1,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmazz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+1,ix, iy)]*c4*g_ex_sigmazz0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+2,ix, iy)]*c3*g_ex_sigmazz0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+3,ix, iy)]*c2*g_ex_sigmazz0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m2m3[n3d_index_ex(iz+4,ix, iy)]*c1*g_ex_sigmazz0_in[n3d_index_ex(iz+4,ix,iy)]
	

									+ g_ex_m2[n3d_index_ex(iz-5,ix, iy)]*c1*g_ex_sigmaxx0_in[n3d_index_ex(iz-5,ix,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz-4,ix, iy)]*c2*g_ex_sigmaxx0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m2[n3d_index_ex(iz-3,ix, iy)]*c3*g_ex_sigmaxx0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-2,ix, iy)]*c4*g_ex_sigmaxx0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-1,ix, iy)]*c5*g_ex_sigmaxx0_in[n3d_index_ex(iz-1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxx0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+1,ix, iy)]*c4*g_ex_sigmaxx0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+2,ix, iy)]*c3*g_ex_sigmaxx0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+3,ix, iy)]*c2*g_ex_sigmaxx0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+4,ix, iy)]*c1*g_ex_sigmaxx0_in[n3d_index_ex(iz+4,ix,iy)]
	

									+ g_ex_m2[n3d_index_ex(iz-5,ix, iy)]*c1*g_ex_sigmayy0_in[n3d_index_ex(iz-5,ix,iy)]							
							 		+ g_ex_m2[n3d_index_ex(iz-4,ix, iy)]*c2*g_ex_sigmayy0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m2[n3d_index_ex(iz-3,ix, iy)]*c3*g_ex_sigmayy0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-2,ix, iy)]*c4*g_ex_sigmayy0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m2[n3d_index_ex(iz-1,ix, iy)]*c5*g_ex_sigmayy0_in[n3d_index_ex(iz-1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+1,ix, iy)]*c4*g_ex_sigmayy0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+2,ix, iy)]*c3*g_ex_sigmayy0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+3,ix, iy)]*c2*g_ex_sigmayy0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m2[n3d_index_ex(iz+4,ix, iy)]*c1*g_ex_sigmayy0_in[n3d_index_ex(iz+4,ix,iy)]
	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m3[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_sigmayz0_in[n3d_index_ex(iz,ix,iy+5)]							
	

									+ g_ex_m3[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m3[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m3[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m3[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_sigmaxz0_in[n3d_index_ex(iz,ix+5,iy)]	;						


		

              g_ex_sigmaxx0_out[n3d_index_ex(iz,ix  ,iy)] =/* g_ex_sigmaxx0_out[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmaxx0_in1[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_x[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_Vx0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m1_x[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_Vx0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_Vx0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_Vx0_in[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_Vx0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_Vx0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_Vx0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_Vx0_in[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_Vx0_in[n3d_index_ex(iz,ix+5,iy)]	;						

	    
              g_ex_sigmayy0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmayy0_out[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmayy0_in1[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy+5)]	;		


              g_ex_sigmazz0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmazz0_out[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmazz0_in1[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_z[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_Vz0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m1_z[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_Vz0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_Vz0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_Vz0_in[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_Vz0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_Vz0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_Vz0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_Vz0_in[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_Vz0_in[n3d_index_ex(iz+5,ix,iy)]	;		 
	
	


              g_ex_sigmaxy0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmaxy0_out[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmaxy0_in1[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_y[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_Vy0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m1_y[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_Vy0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_Vy0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_Vy0_in[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_Vy0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_Vy0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_Vy0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_Vy0_in[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_Vy0_in[n3d_index_ex(iz,ix+5,iy)]	

	    
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m1_x[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy+5)]	;		


              g_ex_sigmaxz0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmaxz0_out[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmaxz0_in1[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_x[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_Vx0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m1_x[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_Vx0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_Vx0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_Vx0_in[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m1_x[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vx0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_Vx0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_Vx0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_Vx0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_Vx0_in[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m1_x[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_Vx0_in[n3d_index_ex(iz+5,ix,iy)]	
							
									+ g_ex_m1_z[n3d_index_ex(iz,ix-4, iy)]*c1*g_ex_Vz0_in[n3d_index_ex(iz,ix-4,iy)]		
									+ g_ex_m1_z[n3d_index_ex(iz,ix-3, iy)]*c2*g_ex_Vz0_in[n3d_index_ex(iz,ix-3,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix-2, iy)]*c3*g_ex_Vz0_in[n3d_index_ex(iz,ix-2,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix-1, iy)]*c4*g_ex_Vz0_in[n3d_index_ex(iz,ix-1,iy)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+1, iy)]*c5*g_ex_Vz0_in[n3d_index_ex(iz,ix+1,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+2, iy)]*c4*g_ex_Vz0_in[n3d_index_ex(iz,ix+2,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+3, iy)]*c3*g_ex_Vz0_in[n3d_index_ex(iz,ix+3,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+4, iy)]*c2*g_ex_Vz0_in[n3d_index_ex(iz,ix+4,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix+5, iy)]*c1*g_ex_Vz0_in[n3d_index_ex(iz,ix+5,iy)]	;						


              g_ex_sigmayz0_out[n3d_index_ex(iz,ix  ,iy)] = /*g_ex_sigmayz0_out[n3d_index_ex(iz,ix  , iy)]*/	+ g_ex_sigmayz0_in1[n3d_index_ex(iz,ix  , iy)] 
									+ g_ex_m1_y[n3d_index_ex(iz-4,ix, iy)]*c1*g_ex_Vy0_in[n3d_index_ex(iz-4,ix,iy)]		
									+ g_ex_m1_y[n3d_index_ex(iz-3,ix, iy)]*c2*g_ex_Vy0_in[n3d_index_ex(iz-3,ix,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz-2,ix, iy)]*c3*g_ex_Vy0_in[n3d_index_ex(iz-2,ix,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz-1,ix, iy)]*c4*g_ex_Vy0_in[n3d_index_ex(iz-1,ix,iy)]	
									+ g_ex_m1_y[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vy0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+1,ix, iy)]*c5*g_ex_Vy0_in[n3d_index_ex(iz+1,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+2,ix, iy)]*c4*g_ex_Vy0_in[n3d_index_ex(iz+2,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+3,ix, iy)]*c3*g_ex_Vy0_in[n3d_index_ex(iz+3,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+4,ix, iy)]*c2*g_ex_Vy0_in[n3d_index_ex(iz+4,ix,iy)]	
									- g_ex_m1_y[n3d_index_ex(iz+5,ix, iy)]*c1*g_ex_Vy0_in[n3d_index_ex(iz+5,ix,iy)]	
	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-4)]*c1*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy-4)]		
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-3)]*c2*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy-3)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-2)]*c3*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy-2)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy-1)]*c4*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy-1)]	
									+ g_ex_m1_z[n3d_index_ex(iz,ix, iy)]  *c5*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+1)]*c5*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy+1)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+2)]*c4*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy+2)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+3)]*c3*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy+3)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+4)]*c2*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy+4)]	
									- g_ex_m1_z[n3d_index_ex(iz,ix, iy+5)]*c1*g_ex_Vz0_in[n3d_index_ex(iz,ix,iy+5)]	;		

}


extern "C" void rtm_gpu_func(int ny, int nz, int nx, 
        float *ex_Vy0_in,  float * ex_Vx0_in, float * ex_Vz0_in, float * ex_sigmayy0_in, float *ex_sigmaxx0_in, float * ex_sigmazz0_in, float * ex_sigmaxy0_in, float * ex_sigmaxz0_in, float * ex_sigmayz0_in,//(nz, nx, nt)
        float *ex_Vy0_in1,  float * ex_Vx0_in1, float * ex_Vz0_in1, float * ex_sigmayy0_in1, float *ex_sigmaxx0_in1, float * ex_sigmazz0_in1, float * ex_sigmaxy0_in1, float * ex_sigmaxz0_in1, float * ex_sigmayz0_in1,//(nz, nx, nt)
        float *ex_Vy0_out,  float * ex_Vx0_out, float * ex_Vz0_out, float * ex_sigmayy0_out, float *ex_sigmaxx0_out, float * ex_sigmazz0_out, float * ex_sigmaxy0_out, float * ex_sigmaxz0_out, float * ex_sigmayz0_out,//(nz, nx, nt)
        float * ex_m1_y, float * ex_m1_x,float * ex_m1_z,float * ex_m2, float * ex_m3, float * ex_m2m3,//)//(nz+10,nx+10)
	float * debug, float * gpu_kernel_time)
{	
     	cudaError_t err;
	cudaEvent_t start1, start2, start3, stop1, stop2, stop3;
	float elapsedTime1 = 0.0f;
	float elapsedTime2 = 0.0f;
	float elapsedTime3 = 0.0f;
	int g_it;

	//config multiple GPU status
	//Acquire the number of GPU available
#ifdef MULTI_GPUS
	int n_gpus = 4;
	int gpu_n;
	cudaGetDeviceCount(&gpu_n);

	if(n_gpus>gpu_n){
		fprintf(stderr, "[EXIT] GPU number avalable %d less than number assigned %d\n", gpu_n, n_gpus);
		//fprintf(stderr, "Set GPU number from assigned %d to GPU number avalable %d \n", n_gpus, gpu_n);
		//n_gpus = gpu_n;
		exit(1);
	}
	setup_cuda(n_gpus);


	float *g_ex_Vx0_in[n_gpus], *g_ex_Vx0_in1[n_gpus], *g_ex_Vx0_out[n_gpus];
	float *g_ex_Vy0_in[n_gpus], *g_ex_Vy0_in1[n_gpus], *g_ex_Vy0_out[n_gpus];
	float *g_ex_Vz0_in[n_gpus], *g_ex_Vz0_in1[n_gpus], *g_ex_Vz0_out[n_gpus];
	float *g_ex_sigmaxx0_in[n_gpus], *g_ex_sigmaxx0_in1[n_gpus], *g_ex_sigmaxx0_out[n_gpus];
	float *g_ex_sigmayy0_in[n_gpus], *g_ex_sigmayy0_in1[n_gpus], *g_ex_sigmayy0_out[n_gpus];
	float *g_ex_sigmazz0_in[n_gpus], *g_ex_sigmazz0_in1[n_gpus], *g_ex_sigmazz0_out[n_gpus];
	float *g_ex_sigmaxy0_in[n_gpus], *g_ex_sigmaxy0_in1[n_gpus], *g_ex_sigmaxy0_out[n_gpus];
	float *g_ex_sigmaxz0_in[n_gpus], *g_ex_sigmaxz0_in1[n_gpus], *g_ex_sigmaxz0_out[n_gpus];
	float *g_ex_sigmayz0_in[n_gpus], *g_ex_sigmayz0_in1[n_gpus], *g_ex_sigmayz0_out[n_gpus];


	float *g_ex_m2[n_gpus];
	float *g_ex_m3[n_gpus];
	float *g_ex_m2m3[n_gpus];
	float *g_ex_m1_x[n_gpus];
	float *g_ex_m1_z[n_gpus];
	float *g_ex_m1_y[n_gpus];

	

	int n1 = (NZ+10), n2 = (NX+10), n3 = (NY+10); // NZ --> NY: fast --> slow, NX, NY, NZ refers to the original index from Fortran code, and here need to be extended by plus 10 for stencil doundary
	n3 = (n3 - 2*radius)/n_gpus + 2*radius;		
	int n_bytes_gpu = (n1*n2*n3)*sizeof(float);
	
	//fprintf(stderr, "LLLLLLLLLLLLLLLLLLLLLLLLLLLLL %d\n", n3);

	//malloc data 
	for(int i=0; i<n_gpus; i++){
	
		cudaSetDevice(device[i]);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaMalloc((void **) &g_ex_Vx0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vx0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vx0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_Vy0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vy0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vy0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_Vz0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vz0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_Vz0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmaxx0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxx0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxx0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmayy0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayy0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayy0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmazz0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmazz0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmazz0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmaxy0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxy0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxy0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmaxz0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxz0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmaxz0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_sigmayz0_in[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayz0_in1[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_sigmayz0_out[i], n_bytes_gpu);

		cudaMalloc((void **) &g_ex_m2[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m3[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m2m3[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m1_x[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m1_y[i], n_bytes_gpu);
		cudaMalloc((void **) &g_ex_m1_z[i], n_bytes_gpu);
		
	}


	int start[n_gpus];	//index for copy data to GPU
	int offset_internal = radius*(n1*n2);
	int n_bytes_gpu_internal = (n3-4*radius)*(n1*n2)*sizeof(float);

	dim3 dimGrid_internal(n1/TZ, n2/TX, (n3-4*radius)/TY);
	dim3 dimBlock(TZ, TX, TY);

	int offset_halo_up = 0;//radius*(n1*n2);	
	int offset_halo_bt = (n3-3*radius)*(n1*n2);
	int n_bytes_gpu_halo = radius*(n1*n2)*sizeof(float);

	dim3 dimGrid_halo(n1/TZ, n2/TX, radius/TY);

	int offset_out = radius * n1*n2;
	int n_bytes_gpu_back = n1*n2*(n3-2*radius)*sizeof(float);
	
	//Define coordinates and offsets parameters
	for(int i=0; i<n_gpus; i++){
		start[i] = i*(n3-2*radius) * (n1*n2);
	}

	fprintf(stderr, "Copy To GPU...\n");

	for(int i = 0; i<n_gpus; i++){
		fprintf(stderr, "Copy to GPU [%d} ...\n", device[i]);
		cudaSetDevice(device[i]);

		cudaMemcpy(g_ex_Vx0_in[i],  &ex_Vx0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vx0_in1[i], &ex_Vx0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vx0_out[i], &ex_Vx0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_Vy0_in[i],  &ex_Vy0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vy0_in1[i], &ex_Vy0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vy0_out[i], &ex_Vy0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_Vz0_in[i],  &ex_Vz0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vz0_in1[i], &ex_Vz0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_Vz0_out[i], &ex_Vz0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmaxx0_in[i], &ex_sigmaxx0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxx0_in1[i], &ex_sigmaxx0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxx0_out[i], &ex_sigmaxx0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmayy0_in[i], &ex_sigmayy0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayy0_in1[i], &ex_sigmayy0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayy0_out[i], &ex_sigmayy0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmazz0_in[i], &ex_sigmazz0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmazz0_in1[i], &ex_sigmazz0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmazz0_out[i], &ex_sigmazz0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmaxy0_in[i], &ex_sigmaxy0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxy0_in1[i], &ex_sigmaxy0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxy0_out[i], &ex_sigmaxy0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmaxz0_in[i], &ex_sigmaxz0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxz0_in1[i], &ex_sigmaxz0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmaxz0_out[i], &ex_sigmaxz0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_sigmayz0_in[i], &ex_sigmayz0_in[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayz0_in1[i], &ex_sigmayz0_in1[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_sigmayz0_out[i], &ex_sigmayz0_out[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

		cudaMemcpy(g_ex_m2[i], &ex_m2[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m3[i], &ex_m3[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m2m3[i], &ex_m2m3[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m1_x[i], &ex_m1_x[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m1_y[i], &ex_m1_y[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);
		cudaMemcpy(g_ex_m1_z[i], &ex_m1_z[start[i]], n_bytes_gpu, cudaMemcpyHostToDevice);

	}
	
	//Define separate streams for overlapping communication
	cudaStream_t stream_halo[n_gpus], stream_internal[n_gpus];
	
	for(int i=0; i<n_gpus; i++){
		cudaSetDevice(device[i]);
		cudaStreamCreate(&stream_halo[i]);
		cudaStreamCreate(&stream_internal[i]);
	}
	fprintf(stderr,"GPU Computing ... ...(NZ=%d, NX=%d, NY=%d, TZ=%d, TX=%d, TY=%d)\n", nz, nx, ny, TZ, TX, TY);

	for(g_it=0; g_it<Steps_write_back;g_it++){
	
	//Calculate the halo regions first
	for(int i=0; i<n_gpus; i++){
		cudaSetDevice(i);
		
      	//if(i>0){
		fprintf(stderr, "Halo computing, GPU id %d\n", i);
		rtm_gpu_kernel<<<dimGrid_halo, dimBlock,0,stream_halo[i]>>>(ny, nz, nx,
			g_ex_Vy0_in[i] + offset_halo_up, g_ex_Vx0_in[i] + offset_halo_up, g_ex_Vz0_in[i] + offset_halo_up, g_ex_sigmayy0_in[i] + offset_halo_up, g_ex_sigmaxx0_in[i] + offset_halo_up, g_ex_sigmazz0_in[i] + offset_halo_up, g_ex_sigmaxy0_in[i] + offset_halo_up, g_ex_sigmaxz0_in[i] + offset_halo_up, g_ex_sigmayz0_in[i] + offset_halo_up,
			g_ex_Vy0_in1[i] + offset_halo_up, g_ex_Vx0_in1[i] + offset_halo_up, g_ex_Vz0_in1[i] + offset_halo_up, g_ex_sigmayy0_in1[i] + offset_halo_up, g_ex_sigmaxx0_in1[i] + offset_halo_up, g_ex_sigmazz0_in1[i] + offset_halo_up, g_ex_sigmaxy0_in1[i] + offset_halo_up, g_ex_sigmaxz0_in1[i] + offset_halo_up, g_ex_sigmayz0_in1[i] + offset_halo_up,
			g_ex_Vy0_out[i] + offset_halo_up, g_ex_Vx0_out[i] + offset_halo_up, g_ex_Vz0_out[i] + offset_halo_up, g_ex_sigmayy0_out[i] + offset_halo_up, g_ex_sigmaxx0_out[i] + offset_halo_up, g_ex_sigmazz0_out[i] + offset_halo_up, g_ex_sigmaxy0_out[i] + offset_halo_up, g_ex_sigmaxz0_out[i] + offset_halo_up, g_ex_sigmayz0_out[i] + offset_halo_up,
			g_ex_m1_y[i] + offset_halo_up, g_ex_m1_x[i] + offset_halo_up, g_ex_m1_z[i] + offset_halo_up, g_ex_m2[i] + offset_halo_up, g_ex_m3[i] + offset_halo_up, g_ex_m2m3[i] + offset_halo_up);
		//}

		//if(i<n_gpus-1){
		fprintf(stderr, "Halo computing, GPU id %d\n", i);
		rtm_gpu_kernel<<<dimGrid_halo, dimBlock,0,stream_halo[i]>>>(ny, nz, nx,
			g_ex_Vy0_in[i] + offset_halo_bt, g_ex_Vx0_in[i] + offset_halo_bt, g_ex_Vz0_in[i] + offset_halo_bt, g_ex_sigmayy0_in[i] + offset_halo_bt, g_ex_sigmaxx0_in[i] + offset_halo_bt, g_ex_sigmazz0_in[i] + offset_halo_bt, g_ex_sigmaxy0_in[i] + offset_halo_bt, g_ex_sigmaxz0_in[i] + offset_halo_bt, g_ex_sigmayz0_in[i] + offset_halo_bt,
			g_ex_Vy0_in1[i] + offset_halo_bt, g_ex_Vx0_in1[i] + offset_halo_bt, g_ex_Vz0_in1[i] + offset_halo_bt, g_ex_sigmayy0_in1[i] + offset_halo_bt, g_ex_sigmaxx0_in1[i] + offset_halo_bt, g_ex_sigmazz0_in1[i] + offset_halo_bt, g_ex_sigmaxy0_in1[i] + offset_halo_bt, g_ex_sigmaxz0_in1[i] + offset_halo_bt, g_ex_sigmayz0_in1[i] + offset_halo_bt,
			g_ex_Vy0_out[i] + offset_halo_bt, g_ex_Vx0_out[i] + offset_halo_bt, g_ex_Vz0_out[i] + offset_halo_bt, g_ex_sigmayy0_out[i] + offset_halo_bt, g_ex_sigmaxx0_out[i] + offset_halo_bt, g_ex_sigmazz0_out[i] + offset_halo_bt, g_ex_sigmaxy0_out[i] + offset_halo_bt, g_ex_sigmaxz0_out[i] + offset_halo_bt, g_ex_sigmayz0_out[i] + offset_halo_bt,
			g_ex_m1_y[i] + offset_halo_bt, g_ex_m1_x[i] + offset_halo_bt, g_ex_m1_z[i] + offset_halo_bt, g_ex_m2[i] + offset_halo_bt, g_ex_m3[i] + offset_halo_bt, g_ex_m2m3[i] + offset_halo_bt);

		cudaStreamQuery(stream_halo[i]);
		//}
	}	

	//Compute the internal part
	for(int i=0; i<n_gpus; i++){

		cudaSetDevice(i);	
		fprintf(stderr, "Internal Computing, GPU id %d\n", i);
		rtm_gpu_kernel<<<dimGrid_internal, dimBlock, 0, stream_internal[i]>>>(ny, nz, nx,
			g_ex_Vy0_in[i] + offset_internal, g_ex_Vx0_in[i] + offset_internal, g_ex_Vz0_in[i] + offset_internal, g_ex_sigmayy0_in[i] + offset_internal, g_ex_sigmaxx0_in[i] + offset_internal, g_ex_sigmazz0_in[i] + offset_internal, g_ex_sigmaxy0_in[i] + offset_internal, g_ex_sigmaxz0_in[i] + offset_internal, g_ex_sigmayz0_in[i] + offset_internal,
			g_ex_Vy0_in1[i] + offset_internal, g_ex_Vx0_in1[i] + offset_internal, g_ex_Vz0_in1[i] + offset_internal, g_ex_sigmayy0_in1[i] + offset_internal, g_ex_sigmaxx0_in1[i] + offset_internal, g_ex_sigmazz0_in1[i] + offset_internal, g_ex_sigmaxy0_in1[i] + offset_internal, g_ex_sigmaxz0_in1[i] + offset_internal, g_ex_sigmayz0_in1[i] + offset_internal,
		g_ex_Vy0_out[i] + offset_internal, g_ex_Vx0_out[i] + offset_internal, g_ex_Vz0_out[i] + offset_internal, g_ex_sigmayy0_out[i] + offset_internal, g_ex_sigmaxx0_out[i] + offset_internal, g_ex_sigmazz0_out[i] + offset_internal, g_ex_sigmaxy0_out[i] + offset_internal, g_ex_sigmaxz0_out[i] + offset_internal, g_ex_sigmayz0_out[i] + offset_internal,
			g_ex_m1_y[i] + offset_internal, g_ex_m1_x[i] + offset_internal, g_ex_m1_z[i] + offset_internal, g_ex_m2[i] + offset_internal, g_ex_m3[i] + offset_internal, g_ex_m2m3[i] + offset_internal);
		}

	//Halo updating

	if(g_it < Steps_write_back-1){

	int offset_down_snd = (n3-2*radius)*(n1*n2);
	int offset_down_rcv = 0;//n3-2*radius)*(n1*n2);
	int offset_up_snd   = radius*n1*n2;
	int offset_up_rcv   = (n3-radius)*n1*n2;
	
	//Send halos downwards
	for(int i=0; i<n_gpus-1; i++){
		cudaMemcpyPeerAsync(g_ex_Vx0_out[i+1]+offset_down_rcv,i+1, g_ex_Vx0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_Vy0_out[i+1]+offset_down_rcv,i+1, g_ex_Vy0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_Vz0_out[i+1]+offset_down_rcv,i+1, g_ex_Vz0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmaxx0_out[i+1]+offset_down_rcv,i+1, g_ex_sigmaxx0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmayy0_out[i+1]+offset_down_rcv,i+1, g_ex_sigmayy0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmazz0_out[i+1]+offset_down_rcv,i+1, g_ex_sigmazz0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmaxy0_out[i+1]+offset_down_rcv,i+1, g_ex_sigmaxy0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmaxz0_out[i+1]+offset_down_rcv,i+1, g_ex_sigmaxz0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmayz0_out[i+1]+offset_down_rcv,i+1, g_ex_sigmayz0_out[i]+offset_down_snd, i, n_bytes_gpu_halo, stream_halo[i]);
	}
	//Synchronize to avoid stalling
	for(int i=0; i<n_gpus; i++){
		cudaSetDevice(i);
		cudaStreamSynchronize(stream_halo[i]);
	}	//Send halos upwards
	//Send halos upwards
	for(int i=1; i<n_gpus; i++){
		cudaMemcpyPeerAsync(g_ex_Vx0_out[i-1]+offset_up_rcv,i-1, g_ex_Vx0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_Vy0_out[i-1]+offset_up_rcv,i-1, g_ex_Vy0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_Vz0_out[i-1]+offset_up_rcv,i-1, g_ex_Vz0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmaxx0_out[i-1]+offset_up_rcv,i-1, g_ex_sigmaxx0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmayy0_out[i-1]+offset_up_rcv,i-1, g_ex_sigmayy0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmazz0_out[i-1]+offset_up_rcv,i-1, g_ex_sigmazz0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmaxz0_out[i-1]+offset_up_rcv,i-1, g_ex_sigmaxz0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmayz0_out[i-1]+offset_up_rcv,i-1, g_ex_sigmayz0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
		cudaMemcpyPeerAsync(g_ex_sigmaxy0_out[i-1]+offset_up_rcv,i-1, g_ex_sigmaxy0_out[i]+offset_up_snd, i, n_bytes_gpu_halo, stream_halo[i]);
	}
	//rtm_gpu_change_pointer(n_gpus);
	//rtm_gpu_change_pointer(n_gpus);
     	//change pointer

		fprintf(stderr, "GPU pointer changed\n");

		float *g_tmp = NULL;

		
		for(int i=0; i<n_gpus; i++){
			cudaSetDevice(device[i]);

			g_tmp = 	g_ex_Vx0_out[i];
			g_ex_Vx0_out[i] = 	g_ex_Vx0_in[i];
			g_ex_Vx0_in[i] = 	g_tmp;

			g_tmp = 	g_ex_Vx0_out[i];
			g_ex_Vx0_out[i] = 	g_ex_Vx0_in1[i];
			g_ex_Vx0_in1[i] = 	g_tmp; 



			g_tmp = 	g_ex_Vy0_out[i];
			g_ex_Vy0_out[i] = 	g_ex_Vy0_in[i];
			g_ex_Vy0_in[i] = 	g_tmp;

			g_tmp = 	g_ex_Vy0_out[i];
			g_ex_Vy0_out[i] = 	g_ex_Vy0_in1[i];
			g_ex_Vy0_in1[i] = 	g_tmp; 



			g_tmp = 	g_ex_Vz0_out[i];
			g_ex_Vz0_out[i] = 	g_ex_Vz0_in[i];
			g_ex_Vz0_in[i] = 	g_tmp;

			g_tmp = 	g_ex_Vz0_out[i];
			g_ex_Vz0_out[i] = 	g_ex_Vz0_in1[i];
			g_ex_Vz0_in1[i] = 	g_tmp; 




			g_tmp = 	g_ex_sigmaxx0_out[i];
			g_ex_sigmaxx0_out[i] = g_ex_sigmaxx0_in[i];
			g_ex_sigmaxx0_in[i] = g_tmp;

			g_tmp = g_ex_sigmaxx0_out[i];
			g_ex_sigmaxx0_out[i] = g_ex_sigmaxx0_in1[i];
			g_ex_sigmaxx0_in1[i] = g_tmp; 



			g_tmp = 	g_ex_sigmayy0_out[i];
			g_ex_sigmayy0_out[i] = g_ex_sigmayy0_in[i];
			g_ex_sigmayy0_in[i] = g_tmp;

			g_tmp = g_ex_sigmayy0_out[i];
			g_ex_sigmayy0_out[i] = g_ex_sigmayy0_in1[i];
			g_ex_sigmayy0_in1[i] = g_tmp; 



			g_tmp = 	g_ex_sigmazz0_out[i];
			g_ex_sigmazz0_out[i] = g_ex_sigmazz0_in[i];
			g_ex_sigmazz0_in[i] = g_tmp;

			g_tmp = g_ex_sigmazz0_out[i];
			g_ex_sigmazz0_out[i] = g_ex_sigmazz0_in1[i];
			g_ex_sigmazz0_in1[i] = g_tmp; 



			g_tmp = 	g_ex_sigmaxy0_out[i];
			g_ex_sigmaxy0_out[i] = g_ex_sigmaxy0_in[i];
			g_ex_sigmaxy0_in[i] = g_tmp;

			g_tmp = g_ex_sigmaxy0_out[i];
			g_ex_sigmaxy0_out[i] = g_ex_sigmaxy0_in1[i];
			g_ex_sigmaxy0_in1[i] = g_tmp; 


			g_tmp = 	g_ex_sigmaxz0_out[i];
			g_ex_sigmaxz0_out[i] = g_ex_sigmaxz0_in[i];
			g_ex_sigmaxz0_in[i] = g_tmp;

			g_tmp = g_ex_sigmaxz0_out[i];
			g_ex_sigmaxz0_out[i] = g_ex_sigmaxz0_in1[i];
			g_ex_sigmaxz0_in1[i] = g_tmp; 


			g_tmp = 	g_ex_sigmayz0_out[i];
			g_ex_sigmayz0_out[i] = g_ex_sigmayz0_in[i];
			g_ex_sigmayz0_in[i] = g_tmp;

			g_tmp = g_ex_sigmayz0_out[i];
			g_ex_sigmayz0_out[i] = g_ex_sigmayz0_in1[i];
			g_ex_sigmayz0_in1[i] = g_tmp; 


		}
	}
	}



	for(int i = 0; i<n_gpus; i++){
		fprintf(stderr, "Copy to CPU [%d} ...\n", device[i]);
		//start_back[i] = (radius + i*(n3-2*radius)) * (n1*n2);
		cudaSetDevice(device[i]);
		//cudaMemcpy(&ex_Vx0_in[start[i]],  g_ex_Vx0_in[i], n_bytes_gpu, cudaMemcpyDeviceToHost);
		//cudaMemcpy(&ex_Vx0_in1[start[i]], g_ex_Vx0_in1[i], n_bytes_gpu, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_Vx0_out[start[i]]+offset_out, g_ex_Vx0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_Vy0_out[start[i]]+offset_out, g_ex_Vy0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_Vz0_out[start[i]]+offset_out, g_ex_Vz0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_sigmaxx0_out[start[i]]+offset_out, g_ex_sigmaxx0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_sigmayy0_out[start[i]]+offset_out, g_ex_sigmayy0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_sigmazz0_out[start[i]]+offset_out, g_ex_sigmazz0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_sigmaxy0_out[start[i]]+offset_out, g_ex_sigmaxy0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_sigmayz0_out[start[i]]+offset_out, g_ex_sigmayz0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);
		cudaMemcpy(&ex_sigmaxz0_out[start[i]]+offset_out, g_ex_sigmaxz0_out[i]+offset_out, n_bytes_gpu_back, cudaMemcpyDeviceToHost);

		//cudaMemcpy(&ex_sigmayy0_in[start[i]], g_ex_sigmayy0_in[i], n_bytes_gpu, cudaMemcpyDeviceToHost);
		//cudaMemcpy(&ex_sigmazz0_in[start[i]], g_ex_sigmazz0_in[i], n_bytes_gpu, cudaMemcpyDeviceToHost);
		//cudaMemcpy(&ex_sigmaxy0_in[start[i]], g_ex_sigmaxy0_in[i], n_bytes_gpu, cudaMemcpyDeviceToHost);
		//cudaMemcpy(&ex_sigmaxz0_in[start[i]], g_ex_sigmaxz0_in[i], n_bytes_gpu, cudaMemcpyDeviceToHost);
		//cudaMemcpy(&ex_sigmayz0_in[start[i]], g_ex_sigmayz0_in[i], n_bytes_gpu, cudaMemcpyDeviceToHost);

	}
	
	
#else
	//-----------blew are old code------------


	rtm_gpu_init(ny,nz,nx, 4);
		

	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);
	cudaEventCreate(&stop3);
	//time record

	//data copy in 
	cudaEventRecord(start1, 0);
     	rtm_gpu_copy_in(ny, nz, nx, 
			ex_Vy0_in, ex_Vx0_in, ex_Vz0_in, ex_sigmayy0_in, ex_sigmaxx0_in, ex_sigmazz0_in, ex_sigmaxy0_in, ex_sigmaxz0_in, ex_sigmayz0_in,
			ex_Vy0_in1, ex_Vx0_in1, ex_Vz0_in1, ex_sigmayy0_in1, ex_sigmaxx0_in1, ex_sigmazz0_in1, ex_sigmaxy0_in1, ex_sigmaxz0_in1, ex_sigmayz0_in1,
			ex_Vy0_out, ex_Vx0_out, ex_Vz0_out, ex_sigmayy0_out, ex_sigmaxx0_out, ex_sigmazz0_out, ex_sigmaxy0_out, ex_sigmaxz0_out, ex_sigmayz0_out,
			ex_m1_y, ex_m1_x, ex_m1_z, ex_m2, ex_m3, ex_m2m3);
	cudaEventRecord(stop1, 0);
	
	
	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error4: %s.\n", cudaGetErrorString(err));
		exit(0);
	}	
	
	//RTM computing


	dim3 dimGrid(nz/TZ, nx/TX, ny/TY);
	dim3 dimBlock(TZ, TX, TY);


	cudaEventRecord(start2, 0);
	
	fprintf(stderr,"GPU Computing ... ...(NZ=%d, NX=%d, NY=%d, TZ=%d, TX=%d, TY=%d)\n", nz, nx, ny, TZ, TX, TY);
	
	for(g_it = 0; g_it < Steps_write_back; g_it++){
		
		fprintf(stderr, "Step %d\n", g_it);
		rtm_gpu_kernel<<<dimGrid, dimBlock>>>(ny, nz, nx,
			g_ex_Vy0_in, g_ex_Vx0_in, g_ex_Vz0_in, g_ex_sigmayy0_in, g_ex_sigmaxx0_in, g_ex_sigmazz0_in, g_ex_sigmaxy0_in, g_ex_sigmaxz0_in, g_ex_sigmayz0_in,
			g_ex_Vy0_in1, g_ex_Vx0_in1, g_ex_Vz0_in1, g_ex_sigmayy0_in1, g_ex_sigmaxx0_in1, g_ex_sigmazz0_in1, g_ex_sigmaxy0_in1, g_ex_sigmaxz0_in1, g_ex_sigmayz0_in1,
			g_ex_Vy0_out, g_ex_Vx0_out, g_ex_Vz0_out, g_ex_sigmayy0_out, g_ex_sigmaxx0_out, g_ex_sigmazz0_out, g_ex_sigmaxy0_out, g_ex_sigmaxz0_out, g_ex_sigmayz0_out,
			g_ex_m1_y, g_ex_m1_x, g_ex_m1_z, g_ex_m2, g_ex_m3, g_ex_m2m3);
			//cudaThreadSynchronize();

		err = cudaGetLastError();
		if(cudaSuccess != err){
			fprintf(stderr, "Cuda error5: %s.\n", cudaGetErrorString(err));
			exit(0);
			}
	
		if(g_it<Steps_write_back-1)	rtm_gpu_change_pointer();	
	}
	cudaEventRecord(stop2, 0);
	

	//data copy out
	cudaEventRecord(start3, 0);
	
	rtm_gpu_copy_out(ny, nz, nx,	
			ex_Vy0_out, ex_Vx0_out, ex_Vz0_out, ex_sigmayy0_out, ex_sigmaxx0_out, ex_sigmazz0_out, ex_sigmaxy0_out, ex_sigmaxz0_out, ex_sigmayz0_out);
	cudaEventRecord(stop3, 0);

	err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error6: %s.\n", cudaGetErrorString(err));
	}	


	//cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop1);
	cudaEventSynchronize(stop2);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&elapsedTime1, start1, stop1);
	cudaEventElapsedTime(&elapsedTime2, start2, stop2);
	cudaEventElapsedTime(&elapsedTime3, start3, stop3);

	gpu_kernel_time[0] = (float)(elapsedTime1/1000.);
	gpu_kernel_time[1] = (float)(elapsedTime2/1000.);
	gpu_kernel_time[2] = (float)(elapsedTime3/1000.);
	
	rtm_gpu_final();

	
	fprintf(stderr, "GPU copy in Time: %.4f\n", (float)elapsedTime1/1000.);
	fprintf(stderr, "GPU Comput. Time: %.4f\n", (float)elapsedTime2/1000.);
	fprintf(stderr, "GPU copy ot Time: %.4f\n", (float)elapsedTime3/1000.);
#endif
}


__global__ void rtm_gpu_kernel_all_shared(int it,int nt, int nz, int nx,
        float * g_ex_Vx0, float * g_ex_Vz0, float * g_ex_sigmaxx0, float * g_ex_sigmazz0, float * g_ex_sigmaxz0, //(nz, nx, nt)
        float * g_ex_m1_x,float * g_ex_m1_z,float * g_ex_aux_m2_c, float * g_ex_aux_m3_c, float * g_ex_aux_m2m3_c)//(nz+10,	nx+10)
{

	float c1=35.0/294912.0,c2=-405.0/229376.0,c3=567.0/40960.0,c4=-735.0/8192.0,c5=19845.0/16384.0;

	//GPU thread index
	int iz, ix;
	iz = blockIdx.x*blockDim.x + threadIdx.x;
	ix = blockIdx.y*blockDim.y + threadIdx.y;
	//gt = it;
 	
	__shared__ float sh_ex_aux_m2m3_c[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_aux_m2_c[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_aux_m3_c[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_m1_x[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_m1_z[(TZ+10)*(TX+10)];


	__shared__ float sh_ex_Vx0[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_Vz0[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_sigmaxx0[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_sigmazz0[(TZ+10)*(TX+10)];
	__shared__ float sh_ex_sigmaxz0[(TZ+10)*(TX+10)];

	//sh_ex_aux_m2m3_c[threadIdx][];

	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_aux_m2m3_c[index_ex(iz,ix)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_aux_m2_c[index_ex(iz,ix)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_aux_m3_c[index_ex(iz,ix)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_m1_x[index_ex(iz,ix)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_m1_z[index_ex(iz,ix)];

	sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_Vx0[index3d_ex(iz,ix,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_Vz0[index3d_ex(iz,ix,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_sigmaxx0[index3d_ex(iz,ix,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_sigmazz0[index3d_ex(iz,ix,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y)] = g_ex_sigmaxz0[index3d_ex(iz,ix,it+1)];


	if(threadIdx.x<5){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_aux_m2m3_c[index_ex(iz-5,ix)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_aux_m2_c[index_ex(iz-5,ix)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_aux_m3_c[index_ex(iz-5,ix)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_m1_x[index_ex(iz-5,ix)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_m1_z[index_ex(iz-5,ix)];

	sh_ex_Vx0[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_Vx0[index3d_ex(iz-5,ix,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_Vz0[index3d_ex(iz-5,ix,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_sigmaxx0[index3d_ex(iz-5,ix,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_sigmazz0[index3d_ex(iz-5,ix,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x-5,threadIdx.y)] = g_ex_sigmaxz0[index3d_ex(iz-5,ix,it+1)];
	}

	if(threadIdx.x>=TZ-5){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_aux_m2m3_c[index_ex(iz+5,ix)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_aux_m2_c[index_ex(iz+5,ix)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_aux_m3_c[index_ex(iz+5,ix)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_m1_x[index_ex(iz+5,ix)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_m1_z[index_ex(iz+5,ix)];
	
	sh_ex_Vx0[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_Vx0[index3d_ex(iz+5,ix,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_Vz0[index3d_ex(iz+5,ix,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_sigmaxx0[index3d_ex(iz+5,ix,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_sigmazz0[index3d_ex(iz+5,ix,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+5,threadIdx.y)] = g_ex_sigmaxz0[index3d_ex(iz+5,ix,it+1)];
	}
	

	if(threadIdx.y<5){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_aux_m2m3_c[index_ex(iz,ix-5)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_aux_m2_c[index_ex(iz,ix-5)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_aux_m3_c[index_ex(iz,ix-5)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_m1_x[index_ex(iz,ix-5)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_m1_z[index_ex(iz,ix-5)];

	sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_Vx0[index3d_ex(iz,ix-5,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_Vz0[index3d_ex(iz,ix-5,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_sigmaxx0[index3d_ex(iz,ix-5,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_sigmazz0[index3d_ex(iz,ix-5,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y-5)] = g_ex_sigmaxz0[index3d_ex(iz,ix-5,it+1)];
	}


	if(threadIdx.y>=TX-5){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_aux_m2m3_c[index_ex(iz,ix+5)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_aux_m2_c[index_ex(iz,ix+5)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_aux_m3_c[index_ex(iz,ix+5)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_m1_x[index_ex(iz,ix+5)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_m1_z[index_ex(iz,ix+5)];

	sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_Vx0[index3d_ex(iz,ix+5,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_Vz0[index3d_ex(iz,ix+5,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_sigmaxx0[index3d_ex(iz,ix+5,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_sigmazz0[index3d_ex(iz,ix+5,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y+5)] = g_ex_sigmaxz0[index3d_ex(iz,ix+5,it+1)];
	}



	if(threadIdx.x <5 && threadIdx.y <5){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_aux_m2m3_c[index_ex(iz-5,ix-5)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_aux_m2_c[index_ex(iz-5,ix-5)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_aux_m3_c[index_ex(iz-5,ix-5)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_m1_x[index_ex(iz-5,ix-5)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_m1_z[index_ex(iz-5,ix-5)];

	sh_ex_Vx0[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_Vx0[index3d_ex(iz-5,ix-5,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_Vz0[index3d_ex(iz-5,ix-5,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_sigmaxx0[index3d_ex(iz-5,ix-5,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_sigmazz0[index3d_ex(iz-5,ix-5,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x-5,threadIdx.y-5)] = g_ex_sigmaxz0[index3d_ex(iz-5,ix-5,it+1)];
	}

	if(threadIdx.x >= 5+TZ && threadIdx.y >= 5+TX){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_aux_m2m3_c[index_ex(iz+5,ix+5)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_aux_m2_c[index_ex(iz+5,ix+5)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_aux_m3_c[index_ex(iz+5,ix+5)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_m1_x[index_ex(iz+5,ix+5)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_m1_z[index_ex(iz+5,ix+5)];

	sh_ex_Vx0[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_Vx0[index3d_ex(iz+5,ix+5,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_Vz0[index3d_ex(iz+5,ix+5,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_sigmaxx0[index3d_ex(iz+5,ix+5,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_sigmazz0[index3d_ex(iz+5,ix+5,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+5,threadIdx.y+5)] = g_ex_sigmaxz0[index3d_ex(iz+5,ix+5,it+1)];
	}


	if(threadIdx.x >= TZ+5 && threadIdx.y <5){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_aux_m2m3_c[index_ex(iz+5,ix-5)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_aux_m2_c[index_ex(iz+5,ix-5)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_aux_m3_c[index_ex(iz+5,ix-5)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_m1_x[index_ex(iz+5,ix-5)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_m1_z[index_ex(iz+5,ix-5)];
	
	sh_ex_Vx0[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_Vx0[index3d_ex(iz+5,ix-5,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_Vz0[index3d_ex(iz+5,ix-5,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_sigmaxx0[index3d_ex(iz+5,ix-5,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_sigmazz0[index3d_ex(iz+5,ix-5,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+5,threadIdx.y-5)] = g_ex_sigmaxz0[index3d_ex(iz+5,ix-5,it+1)];
	}


	if(threadIdx.x <5 && threadIdx.y >= TX-5){
	sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_aux_m2m3_c[index_ex(iz-5,ix+5)];
	sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_aux_m2_c[index_ex(iz-5,ix+5)];
	sh_ex_aux_m3_c[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_aux_m3_c[index_ex(iz-5,ix+5)];
	sh_ex_m1_x[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_m1_x[index_ex(iz-5,ix+5)];
	sh_ex_m1_z[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_m1_z[index_ex(iz-5,ix+5)];

	sh_ex_Vx0[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_Vx0[index3d_ex(iz-5,ix+5,it+1)];
	sh_ex_Vz0[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_Vz0[index3d_ex(iz-5,ix+5,it+1)];
	sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_sigmaxx0[index3d_ex(iz-5,ix+5,it+1)];
	sh_ex_sigmazz0[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_sigmazz0[index3d_ex(iz-5,ix+5,it+1)];
	sh_ex_sigmaxz0[index_blk_ex(threadIdx.x-5,threadIdx.y+5)] = g_ex_sigmaxz0[index3d_ex(iz-5,ix+5,it+1)];
	}



	__syncthreads();

              g_ex_Vx0[index3d_ex(iz,ix  ,it)] = g_ex_Vx0[index3d_ex(iz,ix  ,it)]	+ g_ex_Vx0[index3d_ex(iz, ix, it+2)]
									+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y-5)]*c1*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y-5)]							
							 		+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y-4)]*c2*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y-4)]		
									+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y-3)]*c3*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y-3)]	
									+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y-2)]*c4*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y-2)]	
									+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y-1)]*c5*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y-1)]	
									- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y)]  *c5*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y)]	
									- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y+1)]*c4*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y+1)]	
									- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y+2)]*c3*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y+2)]	
									- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y+3)]*c2*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y+3)]	
									- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,threadIdx.y+4)]*c1*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y+4)]


									+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y-5)]*c1*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y-5)]							
							 		+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y-4)]*c2*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y-4)]		
									+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y-3)]*c3*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y-3)]	
									+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y-2)]*c4*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y-2)]	
									+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y-1)]*c5*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y-1)]	
									- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y)]  *c5*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y)]	
									- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y+1)]*c4*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y+1)]	
									- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y+2)]*c3*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y+2)]	
									- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y+3)]*c2*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y+3)]	
									- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,threadIdx.y+4)]*c1*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y+4)]	
	


									+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x-4,threadIdx.y)]*c1*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x-4,threadIdx.y)]		
									+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x-3,threadIdx.y)]*c2*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x-3,threadIdx.y)]	
									+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x-2,threadIdx.y)]*c3*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x-2,threadIdx.y)]	
									+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x-1,threadIdx.y)]*c4*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x-1,threadIdx.y)]	
									+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,  threadIdx.y)]  *c5*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y)]	
									- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+1,threadIdx.y)]*c5*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+1,threadIdx.y)]	
									- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+2,threadIdx.y)]*c4*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+2,threadIdx.y)]	
									- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+3,threadIdx.y)]*c3*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+3,threadIdx.y)]	
									- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+4,threadIdx.y)]*c2*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+4,threadIdx.y)]	
									- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x+5,threadIdx.y)]*c1*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x+5,threadIdx.y)]	;						

 
     __syncthreads();

            g_ex_Vz0[index3d_ex(iz,ix  ,it)] = g_ex_Vz0[index3d_ex(iz,ix,  it)]  	+ g_ex_Vz0[index3d_ex(iz,ix  ,it+2)] 
	     								+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-5,threadIdx.y)]*c1*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-5,threadIdx.y)]							
	     						 		+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-4,threadIdx.y)]*c2*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-4,threadIdx.y)]		
	     								+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-3,threadIdx.y)]*c3*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-3,threadIdx.y)]	
	     								+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-2,threadIdx.y)]*c4*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-2,threadIdx.y)]	
	     								+ sh_ex_aux_m2_c[index_blk_ex(threadIdx.x-1,threadIdx.y)]*c5*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x-1,threadIdx.y)]	
	     								- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x,  threadIdx.y)]  *c5*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x,threadIdx.y)]	
	     								- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x+1,threadIdx.y)]*c4*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x+1,threadIdx.y)]	
	     								- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x+2,threadIdx.y)]*c3*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x+2,threadIdx.y)]	
	     								- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x+3,threadIdx.y)]*c2*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x+3,threadIdx.y)]	
	     								- sh_ex_aux_m2_c[index_blk_ex(threadIdx.x+4,threadIdx.y)]*c1*sh_ex_sigmaxx0[index_blk_ex(threadIdx.x+4,threadIdx.y)]	
	     
	
	             							+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-5,threadIdx.y)]*c1*sh_ex_sigmazz0[index_blk_ex(threadIdx.x-5,threadIdx.y)]							
	     						 		+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-4,threadIdx.y)]*c2*sh_ex_sigmazz0[index_blk_ex(threadIdx.x-4,threadIdx.y)]		
	     								+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-3,threadIdx.y)]*c3*sh_ex_sigmazz0[index_blk_ex(threadIdx.x-3,threadIdx.y)]	
	     								+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-2,threadIdx.y)]*c4*sh_ex_sigmazz0[index_blk_ex(threadIdx.x-2,threadIdx.y)]	
	     								+ sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x-1,threadIdx.y)]*c5*sh_ex_sigmazz0[index_blk_ex(threadIdx.x-1,threadIdx.y)]	
	     								- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x,  threadIdx.y)]  *c5*sh_ex_sigmazz0[index_blk_ex(threadIdx.x,threadIdx.y)]	
	     								- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x+1,threadIdx.y)]*c4*sh_ex_sigmazz0[index_blk_ex(threadIdx.x+1,threadIdx.y)]	
	     								- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x+2,threadIdx.y)]*c3*sh_ex_sigmazz0[index_blk_ex(threadIdx.x+2,threadIdx.y)]	
	     								- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x+3,threadIdx.y)]*c2*sh_ex_sigmazz0[index_blk_ex(threadIdx.x+3,threadIdx.y)]	
	     								- sh_ex_aux_m2m3_c[index_blk_ex(threadIdx.x+4,threadIdx.y)]*c1*sh_ex_sigmazz0[index_blk_ex(threadIdx.x+4,threadIdx.y)]	
	     
	     								+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y-4)]*c1*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y-4)]		
	     								+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y-3)]*c2*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y-3)]	
	     								+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y-2)]*c3*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y-2)]	
	     								+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y-1)]*c4*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y-1)]	
	     								+ sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y)]  *c5*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y)]	
	     								- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y+1)]*c5*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y+1)]	
	     								- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y+2)]*c4*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y+2)]	
	     								- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y+3)]*c3*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y+3)]	
	     								- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y+4)]*c2*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y+4)]	
	     								- sh_ex_aux_m3_c[index_blk_ex(threadIdx.x,threadIdx.y+5)]*c1*sh_ex_sigmaxz0[index_blk_ex(threadIdx.x,threadIdx.y+5)]	;							
	


              g_ex_sigmaxx0[index3d_ex(iz,ix  ,it)] = g_ex_sigmaxx0[index3d_ex(iz,ix  ,it)]	+ g_ex_sigmaxx0[index3d_ex(iz,ix  ,it+2)] 
        									+ sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y-4)]*c1*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y-4)]		
        									+ sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y-3)]*c2*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y-3)]	
        									+ sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y-2)]*c3*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y-2)]	
        									+ sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y-1)]*c4*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y-1)]	
        									+ sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y)]  *c5*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y)]	
        									- sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y+1)]*c5*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y+1)]	
        									- sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y+2)]*c4*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y+2)]	
        									- sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y+3)]*c3*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y+3)]	
        									- sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y+4)]*c2*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y+4)]	
        									- sh_ex_m1_x[index_blk_ex(threadIdx.x,threadIdx.y+5)]*c1*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y+5)]	;						
 
    __syncthreads();
             g_ex_sigmazz0[index3d_ex(iz,ix  ,it)] = g_ex_sigmazz0[index3d_ex(iz,ix  ,it)]	+ g_ex_sigmazz0[index3d_ex(iz,ix  ,it+2)] 
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x-4,threadIdx.y)]*c1*sh_ex_Vz0[index_blk_ex(threadIdx.x-4,threadIdx.y)]		
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x-3,threadIdx.y)]*c2*sh_ex_Vz0[index_blk_ex(threadIdx.x-3,threadIdx.y)]	
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x-2,threadIdx.y)]*c3*sh_ex_Vz0[index_blk_ex(threadIdx.x-2,threadIdx.y)]	
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x-1,threadIdx.y)]*c4*sh_ex_Vz0[index_blk_ex(threadIdx.x-1,threadIdx.y)]	
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x,  threadIdx.y)]  *c5*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x+1,threadIdx.y)]*c5*sh_ex_Vz0[index_blk_ex(threadIdx.x+1,threadIdx.y)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x+2,threadIdx.y)]*c4*sh_ex_Vz0[index_blk_ex(threadIdx.x+2,threadIdx.y)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x+3,threadIdx.y)]*c3*sh_ex_Vz0[index_blk_ex(threadIdx.x+3,threadIdx.y)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x+4,threadIdx.y)]*c2*sh_ex_Vz0[index_blk_ex(threadIdx.x+4,threadIdx.y)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x+5,threadIdx.y)]*c1*sh_ex_Vz0[index_blk_ex(threadIdx.x+5,threadIdx.y)]	;						
     __syncthreads();
     g_ex_sigmaxz0[index3d_ex(iz,ix  ,it)] = g_ex_sigmaxz0[index3d_ex(iz,ix  ,it)]	+ g_ex_sigmaxz0[index3d_ex(iz,ix  ,it+2)]	 
										+ sh_ex_m1_x[index_blk_ex(threadIdx.x-5,threadIdx.y)]*c1*sh_ex_Vx0[index_blk_ex(threadIdx.x-5,threadIdx.y)]							
							 			+ sh_ex_m1_x[index_blk_ex(threadIdx.x-4,threadIdx.y)]*c2*sh_ex_Vx0[index_blk_ex(threadIdx.x-4,threadIdx.y)]		
										+ sh_ex_m1_x[index_blk_ex(threadIdx.x-3,threadIdx.y)]*c3*sh_ex_Vx0[index_blk_ex(threadIdx.x-3,threadIdx.y)]	
										+ sh_ex_m1_x[index_blk_ex(threadIdx.x-2,threadIdx.y)]*c4*sh_ex_Vx0[index_blk_ex(threadIdx.x-2,threadIdx.y)]	
										+ sh_ex_m1_x[index_blk_ex(threadIdx.x-1,threadIdx.y)]*c5*sh_ex_Vx0[index_blk_ex(threadIdx.x-1,threadIdx.y)]	
										- sh_ex_m1_x[index_blk_ex(threadIdx.x,  threadIdx.y)]  *c5*sh_ex_Vx0[index_blk_ex(threadIdx.x,threadIdx.y)]	
										- sh_ex_m1_x[index_blk_ex(threadIdx.x+1,threadIdx.y)]*c4*sh_ex_Vx0[index_blk_ex(threadIdx.x+1,threadIdx.y)]	
										- sh_ex_m1_x[index_blk_ex(threadIdx.x+2,threadIdx.y)]*c3*sh_ex_Vx0[index_blk_ex(threadIdx.x+2,threadIdx.y)]	
										- sh_ex_m1_x[index_blk_ex(threadIdx.x+3,threadIdx.y)]*c2*sh_ex_Vx0[index_blk_ex(threadIdx.x+3,threadIdx.y)]	
										- sh_ex_m1_x[index_blk_ex(threadIdx.x+4,threadIdx.y)]*c1*sh_ex_Vx0[index_blk_ex(threadIdx.x+4,threadIdx.y)]	//;
	
        
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y-5)]*c1*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y-5)]							
							 			+ sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y-4)]*c2*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y-4)]		
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y-3)]*c3*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y-3)]	
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y-2)]*c4*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y-2)]	
										+ sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y-1)]*c5*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y-1)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y)]  *c5*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y+1)]*c4*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y+1)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y+2)]*c3*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y+2)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y+3)]*c2*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y+3)]	
										- sh_ex_m1_z[index_blk_ex(threadIdx.x,threadIdx.y+4)]*c1*sh_ex_Vz0[index_blk_ex(threadIdx.x,threadIdx.y+4)]	;
		
	__syncthreads();


	}


__global__ void rtm_gpu_kernel_l1(int it,int nt, int nz, int nx,
        float * g_ex_Vx0, float * g_ex_Vz0, float * g_ex_sigmaxx0, float * g_ex_sigmazz0, float * g_ex_sigmaxz0, //(nz, nx, nt)
        float * g_ex_m1_x,float * g_ex_m1_z,float * g_ex_aux_m2_c, float * g_ex_aux_m3_c, float * g_ex_aux_m2m3_c)//(nz+10,	nx+10)
{

	float c1=35.0/294912.0,c2=-405.0/229376.0,c3=567.0/40960.0,c4=-735.0/8192.0,c5=19845.0/16384.0;

	//GPU thread index
	int iz, ix;
	iz = blockIdx.x*blockDim.x + threadIdx.x;
	ix = blockIdx.y*blockDim.y + threadIdx.y;
	//gt = it;
 	
              g_ex_Vx0[index3d_ex(iz,ix  ,it)] = g_ex_Vx0[index3d_ex(iz,ix  ,it)]	+ g_ex_Vx0[index3d_ex(iz, ix, it+2)]
									+ g_ex_aux_m2m3_c[index_ex(iz,ix-5)]*c1*g_ex_sigmaxx0[index3d_ex(iz,ix-5,it+1)]							
							 		+ g_ex_aux_m2m3_c[index_ex(iz,ix-4)]*c2*g_ex_sigmaxx0[index3d_ex(iz,ix-4,it+1)]		
									+ g_ex_aux_m2m3_c[index_ex(iz,ix-3)]*c3*g_ex_sigmaxx0[index3d_ex(iz,ix-3,it+1)]	
									+ g_ex_aux_m2m3_c[index_ex(iz,ix-2)]*c4*g_ex_sigmaxx0[index3d_ex(iz,ix-2,it+1)]	
									+ g_ex_aux_m2m3_c[index_ex(iz,ix-1)]*c5*g_ex_sigmaxx0[index3d_ex(iz,ix-1,it+1)]	
									- g_ex_aux_m2m3_c[index_ex(iz,ix)]  *c5*g_ex_sigmaxx0[index3d_ex(iz,ix,it+1)]	
									- g_ex_aux_m2m3_c[index_ex(iz,ix+1)]*c4*g_ex_sigmaxx0[index3d_ex(iz,ix+1,it+1)]	
									- g_ex_aux_m2m3_c[index_ex(iz,ix+2)]*c3*g_ex_sigmaxx0[index3d_ex(iz,ix+2,it+1)]	
									- g_ex_aux_m2m3_c[index_ex(iz,ix+3)]*c2*g_ex_sigmaxx0[index3d_ex(iz,ix+3,it+1)]	
									- g_ex_aux_m2m3_c[index_ex(iz,ix+4)]*c1*g_ex_sigmaxx0[index3d_ex(iz,ix+4,it+1)]


									+ g_ex_aux_m2_c[index_ex(iz,ix-5)]*c1*g_ex_sigmazz0[index3d_ex(iz,ix-5,it+1)]							
							 		+ g_ex_aux_m2_c[index_ex(iz,ix-4)]*c2*g_ex_sigmazz0[index3d_ex(iz,ix-4,it+1)]		
									+ g_ex_aux_m2_c[index_ex(iz,ix-3)]*c3*g_ex_sigmazz0[index3d_ex(iz,ix-3,it+1)]	
									+ g_ex_aux_m2_c[index_ex(iz,ix-2)]*c4*g_ex_sigmazz0[index3d_ex(iz,ix-2,it+1)]	
									+ g_ex_aux_m2_c[index_ex(iz,ix-1)]*c5*g_ex_sigmazz0[index3d_ex(iz,ix-1,it+1)]	
									- g_ex_aux_m2_c[index_ex(iz,ix)]  *c5*g_ex_sigmazz0[index3d_ex(iz,ix,it+1)]	
									- g_ex_aux_m2_c[index_ex(iz,ix+1)]*c4*g_ex_sigmazz0[index3d_ex(iz,ix+1,it+1)]	
									- g_ex_aux_m2_c[index_ex(iz,ix+2)]*c3*g_ex_sigmazz0[index3d_ex(iz,ix+2,it+1)]	
									- g_ex_aux_m2_c[index_ex(iz,ix+3)]*c2*g_ex_sigmazz0[index3d_ex(iz,ix+3,it+1)]	
									- g_ex_aux_m2_c[index_ex(iz,ix+4)]*c1*g_ex_sigmazz0[index3d_ex(iz,ix+4,it+1)]	
	


									+ g_ex_aux_m3_c[index_ex(iz-4,ix)]*c1*g_ex_sigmaxz0[index3d_ex(iz-4,ix,it+1)]		
									+ g_ex_aux_m3_c[index_ex(iz-3,ix)]*c2*g_ex_sigmaxz0[index3d_ex(iz-3,ix,it+1)]	
									+ g_ex_aux_m3_c[index_ex(iz-2,ix)]*c3*g_ex_sigmaxz0[index3d_ex(iz-2,ix,it+1)]	
									+ g_ex_aux_m3_c[index_ex(iz-1,ix)]*c4*g_ex_sigmaxz0[index3d_ex(iz-1,ix,it+1)]	
									+ g_ex_aux_m3_c[index_ex(iz,ix)]  *c5*g_ex_sigmaxz0[index3d_ex(iz,ix,it+1)]	
									- g_ex_aux_m3_c[index_ex(iz+1,ix)]*c5*g_ex_sigmaxz0[index3d_ex(iz+1,ix,it+1)]	
									- g_ex_aux_m3_c[index_ex(iz+2,ix)]*c4*g_ex_sigmaxz0[index3d_ex(iz+2,ix,it+1)]	
									- g_ex_aux_m3_c[index_ex(iz+3,ix)]*c3*g_ex_sigmaxz0[index3d_ex(iz+3,ix,it+1)]	
									- g_ex_aux_m3_c[index_ex(iz+4,ix)]*c2*g_ex_sigmaxz0[index3d_ex(iz+4,ix,it+1)]	
									- g_ex_aux_m3_c[index_ex(iz+5,ix)]*c1*g_ex_sigmaxz0[index3d_ex(iz+5,ix,it+1)]	;						

 

            g_ex_Vz0[index3d_ex(iz,ix  ,it)] = g_ex_Vz0[index3d_ex(iz,ix  ,it)]  	+ g_ex_Vz0[index3d_ex(iz,ix  ,it+2)] 
	     								+ g_ex_aux_m2_c[index_ex(iz-5,ix)]*c1*g_ex_sigmaxx0[index3d_ex(iz-5,ix,it+1)]							
	     						 		+ g_ex_aux_m2_c[index_ex(iz-4,ix)]*c2*g_ex_sigmaxx0[index3d_ex(iz-4,ix,it+1)]		
	     								+ g_ex_aux_m2_c[index_ex(iz-3,ix)]*c3*g_ex_sigmaxx0[index3d_ex(iz-3,ix,it+1)]	
	     								+ g_ex_aux_m2_c[index_ex(iz-2,ix)]*c4*g_ex_sigmaxx0[index3d_ex(iz-2,ix,it+1)]	
	     								+ g_ex_aux_m2_c[index_ex(iz-1,ix)]*c5*g_ex_sigmaxx0[index3d_ex(iz-1,ix,it+1)]	
	     								- g_ex_aux_m2_c[index_ex(iz,ix)]  *c5*g_ex_sigmaxx0[index3d_ex(iz,ix,it+1)]	
	     								- g_ex_aux_m2_c[index_ex(iz+1,ix)]*c4*g_ex_sigmaxx0[index3d_ex(iz+1,ix,it+1)]	
	     								- g_ex_aux_m2_c[index_ex(iz+2,ix)]*c3*g_ex_sigmaxx0[index3d_ex(iz+2,ix,it+1)]	
	     								- g_ex_aux_m2_c[index_ex(iz+3,ix)]*c2*g_ex_sigmaxx0[index3d_ex(iz+3,ix,it+1)]	
	     								- g_ex_aux_m2_c[index_ex(iz+4,ix)]*c1*g_ex_sigmaxx0[index3d_ex(iz+4,ix,it+1)]	
	     
	
	             							+ g_ex_aux_m2m3_c[index_ex(iz-5,ix)]*c1*g_ex_sigmazz0[index3d_ex(iz-5,ix,it+1)]							
	     						 		+ g_ex_aux_m2m3_c[index_ex(iz-4,ix)]*c2*g_ex_sigmazz0[index3d_ex(iz-4,ix,it+1)]		
	     								+ g_ex_aux_m2m3_c[index_ex(iz-3,ix)]*c3*g_ex_sigmazz0[index3d_ex(iz-3,ix,it+1)]	
	     								+ g_ex_aux_m2m3_c[index_ex(iz-2,ix)]*c4*g_ex_sigmazz0[index3d_ex(iz-2,ix,it+1)]	
	     								+ g_ex_aux_m2m3_c[index_ex(iz-1,ix)]*c5*g_ex_sigmazz0[index3d_ex(iz-1,ix,it+1)]	
	     								- g_ex_aux_m2m3_c[index_ex(iz,ix)]  *c5*g_ex_sigmazz0[index3d_ex(iz,ix,it+1)]	
	     								- g_ex_aux_m2m3_c[index_ex(iz+1,ix)]*c4*g_ex_sigmazz0[index3d_ex(iz+1,ix,it+1)]	
	     								- g_ex_aux_m2m3_c[index_ex(iz+2,ix)]*c3*g_ex_sigmazz0[index3d_ex(iz+2,ix,it+1)]	
	     								- g_ex_aux_m2m3_c[index_ex(iz+3,ix)]*c2*g_ex_sigmazz0[index3d_ex(iz+3,ix,it+1)]	
	     								- g_ex_aux_m2m3_c[index_ex(iz+4,ix)]*c1*g_ex_sigmazz0[index3d_ex(iz+4,ix,it+1)]	
	     
	     								+ g_ex_aux_m3_c[index_ex(iz,ix-4)]*c1*g_ex_sigmaxz0[index3d_ex(iz,ix-4,it+1)]		
	     								+ g_ex_aux_m3_c[index_ex(iz,ix-3)]*c2*g_ex_sigmaxz0[index3d_ex(iz,ix-3,it+1)]	
	     								+ g_ex_aux_m3_c[index_ex(iz,ix-2)]*c3*g_ex_sigmaxz0[index3d_ex(iz,ix-2,it+1)]	
	     								+ g_ex_aux_m3_c[index_ex(iz,ix-1)]*c4*g_ex_sigmaxz0[index3d_ex(iz,ix-1,it+1)]	
	     								+ g_ex_aux_m3_c[index_ex(iz,ix)]  *c5*g_ex_sigmaxz0[index3d_ex(iz,ix,it+1)]	
	     								- g_ex_aux_m3_c[index_ex(iz,ix+1)]*c5*g_ex_sigmaxz0[index3d_ex(iz,ix+1,it+1)]	
	     								- g_ex_aux_m3_c[index_ex(iz,ix+2)]*c4*g_ex_sigmaxz0[index3d_ex(iz,ix+2,it+1)]	
	     								- g_ex_aux_m3_c[index_ex(iz,ix+3)]*c3*g_ex_sigmaxz0[index3d_ex(iz,ix+3,it+1)]	
	     								- g_ex_aux_m3_c[index_ex(iz,ix+4)]*c2*g_ex_sigmaxz0[index3d_ex(iz,ix+4,it+1)]	
	     								- g_ex_aux_m3_c[index_ex(iz,ix+5)]*c1*g_ex_sigmaxz0[index3d_ex(iz,ix+5,it+1)]	;							
	


              g_ex_sigmaxx0[index3d_ex(iz,ix  ,it)] = g_ex_sigmaxx0[index3d_ex(iz,ix  ,it)]	+ g_ex_sigmaxx0[index3d_ex(iz,ix  ,it+2)] 
        									+ g_ex_m1_x[index_ex(iz,ix-4)]*c1*g_ex_Vx0[index3d_ex(iz,ix-4,it+1)]		
        									+ g_ex_m1_x[index_ex(iz,ix-3)]*c2*g_ex_Vx0[index3d_ex(iz,ix-3,it+1)]	
        									+ g_ex_m1_x[index_ex(iz,ix-2)]*c3*g_ex_Vx0[index3d_ex(iz,ix-2,it+1)]	
        									+ g_ex_m1_x[index_ex(iz,ix-1)]*c4*g_ex_Vx0[index3d_ex(iz,ix-1,it+1)]	
        									+ g_ex_m1_x[index_ex(iz,ix)]  *c5*g_ex_Vx0[index3d_ex(iz,ix,it+1)]	
        									- g_ex_m1_x[index_ex(iz,ix+1)]*c5*g_ex_Vx0[index3d_ex(iz,ix+1,it+1)]	
        									- g_ex_m1_x[index_ex(iz,ix+2)]*c4*g_ex_Vx0[index3d_ex(iz,ix+2,it+1)]	
        									- g_ex_m1_x[index_ex(iz,ix+3)]*c3*g_ex_Vx0[index3d_ex(iz,ix+3,it+1)]	
        									- g_ex_m1_x[index_ex(iz,ix+4)]*c2*g_ex_Vx0[index3d_ex(iz,ix+4,it+1)]	
        									- g_ex_m1_x[index_ex(iz,ix+5)]*c1*g_ex_Vx0[index3d_ex(iz,ix+5,it+1)]	;						
 
             g_ex_sigmazz0[index3d_ex(iz,ix  ,it)] = g_ex_sigmazz0[index3d_ex(iz,ix  ,it)]	+ g_ex_sigmazz0[index3d_ex(iz,ix  ,it+2)] 
										+ g_ex_m1_z[index_ex(iz-4,ix)]*c1*g_ex_Vz0[index3d_ex(iz-4,ix,it+1)]		
										+ g_ex_m1_z[index_ex(iz-3,ix)]*c2*g_ex_Vz0[index3d_ex(iz-3,ix,it+1)]	
										+ g_ex_m1_z[index_ex(iz-2,ix)]*c3*g_ex_Vz0[index3d_ex(iz-2,ix,it+1)]	
										+ g_ex_m1_z[index_ex(iz-1,ix)]*c4*g_ex_Vz0[index3d_ex(iz-1,ix,it+1)]	
										+ g_ex_m1_z[index_ex(iz,ix)]  *c5*g_ex_Vz0[index3d_ex(iz,ix,it+1)]	
										- g_ex_m1_z[index_ex(iz+1,ix)]*c5*g_ex_Vz0[index3d_ex(iz+1,ix,it+1)]	
										- g_ex_m1_z[index_ex(iz+2,ix)]*c4*g_ex_Vz0[index3d_ex(iz+2,ix,it+1)]	
										- g_ex_m1_z[index_ex(iz+3,ix)]*c3*g_ex_Vz0[index3d_ex(iz+3,ix,it+1)]	
										- g_ex_m1_z[index_ex(iz+4,ix)]*c2*g_ex_Vz0[index3d_ex(iz+4,ix,it+1)]	
										- g_ex_m1_z[index_ex(iz+5,ix)]*c1*g_ex_Vz0[index3d_ex(iz+5,ix,it+1)]	;						
     
	g_ex_sigmaxz0[index3d_ex(iz,ix  ,it)] = g_ex_sigmaxz0[index3d_ex(iz,ix  ,it)]	+ g_ex_sigmaxz0[index3d_ex(iz,ix  ,it+2)]	 
										+ g_ex_m1_x[index_ex(iz-5,ix)]*c1*g_ex_Vx0[index3d_ex(iz-5,ix,it+1)]							
							 			+ g_ex_m1_x[index_ex(iz-4,ix)]*c2*g_ex_Vx0[index3d_ex(iz-4,ix,it+1)]		
										+ g_ex_m1_x[index_ex(iz-3,ix)]*c3*g_ex_Vx0[index3d_ex(iz-3,ix,it+1)]	
										+ g_ex_m1_x[index_ex(iz-2,ix)]*c4*g_ex_Vx0[index3d_ex(iz-2,ix,it+1)]	
										+ g_ex_m1_x[index_ex(iz-1,ix)]*c5*g_ex_Vx0[index3d_ex(iz-1,ix,it+1)]	
										- g_ex_m1_x[index_ex(iz,ix)]  *c5*g_ex_Vx0[index3d_ex(iz,ix,it+1)]	
										- g_ex_m1_x[index_ex(iz+1,ix)]*c4*g_ex_Vx0[index3d_ex(iz+1,ix,it+1)]	
										- g_ex_m1_x[index_ex(iz+2,ix)]*c3*g_ex_Vx0[index3d_ex(iz+2,ix,it+1)]	
										- g_ex_m1_x[index_ex(iz+3,ix)]*c2*g_ex_Vx0[index3d_ex(iz+3,ix,it+1)]	
										- g_ex_m1_x[index_ex(iz+4,ix)]*c1*g_ex_Vx0[index3d_ex(iz+4,ix,it+1)]	//;
	
        
										+ g_ex_m1_z[index_ex(iz,ix-5)]*c1*g_ex_Vz0[index3d_ex(iz,ix-5,it+1)]							
							 			+ g_ex_m1_z[index_ex(iz,ix-4)]*c2*g_ex_Vz0[index3d_ex(iz,ix-4,it+1)]		
										+ g_ex_m1_z[index_ex(iz,ix-3)]*c3*g_ex_Vz0[index3d_ex(iz,ix-3,it+1)]	
										+ g_ex_m1_z[index_ex(iz,ix-2)]*c4*g_ex_Vz0[index3d_ex(iz,ix-2,it+1)]	
										+ g_ex_m1_z[index_ex(iz,ix-1)]*c5*g_ex_Vz0[index3d_ex(iz,ix-1,it+1)]	
										- g_ex_m1_z[index_ex(iz,ix)]  *c5*g_ex_Vz0[index3d_ex(iz,ix,it+1)]	
										- g_ex_m1_z[index_ex(iz,ix+1)]*c4*g_ex_Vz0[index3d_ex(iz,ix+1,it+1)]	
										- g_ex_m1_z[index_ex(iz,ix+2)]*c3*g_ex_Vz0[index3d_ex(iz,ix+2,it+1)]	
										- g_ex_m1_z[index_ex(iz,ix+3)]*c2*g_ex_Vz0[index3d_ex(iz,ix+3,it+1)]	
										- g_ex_m1_z[index_ex(iz,ix+4)]*c1*g_ex_Vz0[index3d_ex(iz,ix+4,it+1)]	;
		

	}



