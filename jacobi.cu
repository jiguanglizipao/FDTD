#include<cuda_runtime.h>
#include<stdio.h>

#define REAL float
#define NX 512 
#define NY 512 
#define NZ 512 
#define T  1000

#define BX 32
#define BY 32
#define BZ 1
#define GZ 16

const float cc = 0.01;
const float ce = 0.02;
const float cw = 0.03;
const float cs = 0.04;
const float cn = 0.05;
const float ct = 0.06;
const float cb = 0.07;

#define dimT 3
#define kDEP 1

#define stencil(curT, curH, curTB)                         \
	if (threadIdx.x > 0 && threadIdx.x < blockDim.x-1 &&   \
		threadIdx.y > 0 && threadIdx.y < blockDim.y-1){    \
		if (global_i > 0 && global_i < nx-1 &&             \
			global_j > 0 && global_j < ny-1){              \
			cur_##curT##_##curH =                          \
				  ce*cur_plane[curTB][lyidx][lxidx+1]      \
				 +cw*cur_plane[curTB][lyidx][lxidx-1]      \
				 +cs*cur_plane[curTB][lyidx+1][lxidx]      \
				 +cn*cur_plane[curTB][lyidx-1][lxidx]      \
				 +ct*cur_##curTB##_2                       \
				 +cb*cur_##curTB##_0                       \
				 +cc*cur_##curTB##_1;                      \
		}else{                                             \
			cur_##curT##_##curH =                          \
				  cur_##curTB##_1;                         \
		}                                                  \
	}

#define stencil_only(curT, curH, curTB)                    \
		cur_##curT##_##curH =                          \
			  ce*(cur_plane[curTB][lyidx][lxidx+1])      \
			 +cw*(cur_plane[curTB][lyidx][lxidx-1])      \
			 +cs*(cur_plane[curTB][lyidx+1][lxidx])      \
			 +cn*(cur_plane[curTB][lyidx-1][lxidx])      \
			 +ct*(cur_##curTB##_2)                      \
			 +cb*(cur_##curTB##_0)                       \
			 +cc*(cur_##curTB##_1);                      
	    

#define write_global_copy(idx_k0, slice, temp_k)           \
		if (threadIdx.x >= dimT &&                         \
			threadIdx.x < blockDim.x-dimT &&               \
			threadIdx.y >= dimT &&                         \
			threadIdx.y < blockDim.y-dimT){                \
			A[idx_k0+temp_k*slice] =                       \
					B[idx_k0+temp_k*slice];                \
		}

#define load_shared_t0_extra(cur_idx)                      \
		if (lyidx == 1)                                    \
			cur_plane[0][0][lxidx] =                       \
				B[cur_idx-nx];                             \
		else if (lyidx == blockDim.y)                      \
			cur_plane[0][blockDim.y+1][lxidx] =            \
				B[cur_idx+nx];                             \
		if (lxidx == 1)                                    \
			cur_plane[0][lyidx][0] =                       \
				B[cur_idx-1];                              \
		else if (lxidx == blockDim.x)                      \
			cur_plane[0][lyidx][blockDim.x+1] =            \
				B[cur_idx+1];                     

#define write_global_cal(idx_k0, slice, temp_k)             \
		if (threadIdx.x >= dimT &&                          \
			threadIdx.x < blockDim.x-dimT &&                \
			threadIdx.y >= dimT &&                          \
			threadIdx.y < blockDim.y-dimT){                 \
			A[idx_k0+temp_k*slice] =                        \
					ce*(cur_plane[dimT-1][lyidx][lxidx+1])  \
				   +cw*(cur_plane[dimT-1][lyidx][lxidx-1])  \
				   +cs*(cur_plane[dimT-1][lyidx+1][lxidx])  \
				   +cn*(cur_plane[dimT-1][lyidx-1][lxidx])  \
				   +ct*(cur_2_2)                            \
				   +cb*(cur_2_0)                            \
				   +cc*(cur_2_1);                           \
		}

__global__ void baseline(REAL* A, REAL* B, int64_t nx, int64_t ny, int64_t nz)
{
	int64_t i = threadIdx.x + blockDim.x*blockIdx.x;
	int64_t j = threadIdx.y + blockDim.y*blockIdx.y;
	int64_t kb = nz/gridDim.z*blockIdx.z;
	int64_t slice = nx*ny;

	int64_t k = kb > 0? kb: 1;
	int64_t ke = (kb+nz/gridDim.z<nz-1)? kb+nz/gridDim.z : nz-1;
	int64_t idx = i + j*nx + k*slice;
	for (; k < ke; k++){
		if (i > 0 && i < nx && j > 0 && j < ny){
			A[idx] = ce*B[idx+1] + cw*B[idx-1] + cs*B[idx+nx] + cn*B[idx-nx]
					+ct*B[idx+slice] + cb*B[idx-slice] + cc*B[idx];
			idx += slice;
		}
	}

	return;
}

__global__ void temporal_blocking(REAL* A, REAL* B, int64_t nx, int64_t ny, int64_t nz)
{
	int64_t global_i = (threadIdx.x-dimT)
				 + (blockDim.x-2*dimT)*blockIdx.x;
	int64_t global_j = (threadIdx.y-dimT) 
				 + (blockDim.y-2*dimT)*blockIdx.y;
	int64_t slice = nx*ny;

	int64_t gidx = global_i + global_j*nx;
	int64_t lxidx = threadIdx.x;
	int64_t lyidx = threadIdx.y;

	/*
	REAL top[dimT][kDEP];
	REAL mid[dimT][1];
	REAL bot[dimT][kDEP];
	REAL cur[dimT][1];
	*/
	//REAL cur[dimT][2*kDEP+2];
	REAL cur_0_0, cur_0_1, cur_0_2, cur_0_3;//from bottom to up
	REAL cur_1_0, cur_1_1, cur_1_2, cur_1_3;
	REAL cur_2_0, cur_2_1, cur_2_2, cur_2_3;
	//int64_t cur_size = 2*kDEP+2;

	__shared__ REAL cur_plane[dimT][BY][BX];

	/////////////////////////////////////////////////////
	//phase1
	/////////////////////////////////////////////////////
	//s1

	if (global_i >= 0 && global_i < nx &&
		global_j >= 0 && global_j < ny){

		cur_0_0 = B[gidx]; 
		cur_1_0 = cur_0_0; cur_2_0 = cur_0_0;

		//if (threadIdx.x >= (dimT-1) && 
		//	threadIdx.x <= blockDim.x-dimT &&
		//	threadIdx.y >= (dimT-1) && 
		//	threadIdx.y <= blockDim.y-dimT){
		//	A[gidx] = B[gidx];
		//}
		write_global_copy(gidx, slice, 0)

		//s2 & s3
		cur_0_1 = B[gidx+slice]; cur_0_2 = B[gidx+2*slice];


		/////////////////////////////////
		//load s2 into cur_plane();
		cur_plane[0][lyidx][lxidx] = cur_0_1;
		//load_shared_t0_extra(gidx+slice)
		__syncthreads();

		/////////////////////////////////////////////////
		//s4~s13
		//s4
		cur_0_3 = B[gidx+3*slice];
		//s5
		stencil(1,1,0)

		//shared_memory update
		__syncthreads();
		cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		cur_plane[0][lyidx][lxidx] = cur_0_1;
		//load_shared_t0_extra(gidx+2*slice)
		__syncthreads();

		//s6
		cur_0_3 = B[gidx+4*slice];
		//s7
		stencil(1,2,0)

		//shared_memory update
		__syncthreads();
		cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		cur_plane[0][lyidx][lxidx] = cur_0_1;
		//load_shared_t0_extra(gidx+3*slice)
		cur_plane[1][lyidx][lxidx] = cur_1_1;
		__syncthreads();

		//s8
		cur_0_3 = B[gidx+5*slice];
		//s9
		stencil(1,3,0)
		//s10
		stencil(2,1,1)
		//shared memory update
		//update s6,s7
		__syncthreads();
		cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		cur_plane[0][lyidx][lxidx] = cur_0_1;
		//load_shared_t0_extra(gidx+4*slice)
		cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
		cur_plane[1][lyidx][lxidx] = cur_1_1;
		__syncthreads();

		//s11
		cur_0_3 = B[gidx+6*slice];
		//s12
		stencil(1,3,0)
		//s13
		stencil(2,2,1)

		//shared memory update
		__syncthreads();
		//update s8
		cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		cur_plane[0][lyidx][lxidx] = cur_0_1;
		//load_shared_t0_extra(gidx+5*slice)
		//update s9
		cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
		cur_plane[1][lyidx][lxidx] = cur_1_1;
		//update s10
		cur_plane[2][lyidx][lxidx] = cur_2_1;
		__syncthreads();



		////check s13
		//if (threadIdx.x >= (dimT-1) && 
		//	threadIdx.x <= blockDim.x-dimT &&
		//	threadIdx.y >= (dimT-1) && 
		//	threadIdx.y <= blockDim.y-dimT){

		//	for (int64_t temp_k = 2; temp_k <= 2 ; temp_k++){
		//		//A[gidx+temp_k*slice] = B[gidx+temp_k*slice];
		//		if (global_i > 0 && global_i < nx-1 &&
		//			global_j > 0 && global_j < ny-1){
		//		//A[gidx+temp_k*slice] = cur[2][temp_k%cur_size];
		//			A[gidx+temp_k*slice] = cur_2_2;
		//		}
		//	}
		//}

		/////////////////////////////////////////////////////
		//phase2
		/////////////////////////////////////////////////////
		//now focus on the index of t=dimT
		for (int64_t temp_k = 1; temp_k < nz-2*dimT; temp_k++){
			//load t=0...
			//for t=0, the k index of buffer loading is
			int64_t k_index = 2*dimT+temp_k;
			cur_0_3 = B[gidx+k_index*slice];

			if (threadIdx.x > 0 && threadIdx.x < blockDim.x-1 &&
				threadIdx.y > 0 && threadIdx.y < blockDim.y-1){
				if (global_i > 0 && global_i < nx-1 &&
					global_j > 0 && global_j < ny-1){
					stencil_only(1,3,0)
					stencil_only(2,3,1)
					write_global_cal(gidx,slice,temp_k)

				}else{
					cur_1_3 = cur_0_1;
					cur_2_3 = cur_1_1;
				}
			}

			//update shared-memory buffer=>cur_plane
			__syncthreads();
			cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
			cur_plane[0][lyidx][lxidx] = cur_0_1;
			//load_shared_t0_extra(gidx+(k_index-1)*slice)

			cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
			cur_plane[1][lyidx][lxidx] = cur_1_1;
			cur_2_0 = cur_2_1; cur_2_1 = cur_2_2; cur_2_2 = cur_2_3;
			cur_plane[2][lyidx][lxidx] = cur_2_1;
			__syncthreads();
		}

		/////////////////////////////////////////////////////
		//phase3
		/////////////////////////////////////////////////////
		int64_t temp_k = nz-6;
		if (threadIdx.x > 0 && threadIdx.x < blockDim.x-1 &&
			threadIdx.y > 0 && threadIdx.y < blockDim.y-1){
			if (global_i > 0 && global_i < nx-1 &&
				global_j > 0 && global_j < ny-1){
				stencil_only(1,3,0)
				stencil_only(2,3,1)
				write_global_cal(gidx, slice, temp_k)
			}else{
				cur_1_3 = cur_0_1;	
				cur_2_3 = cur_1_1;
			}
		}

		__syncthreads();
		//cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		//cur_plane[0][lyidx][lxidx] = cur_0_1;
		cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
		cur_plane[1][lyidx][lxidx] = cur_1_1;
		cur_2_0 = cur_2_1; cur_2_1 = cur_2_2; cur_2_2 = cur_2_3;
		cur_plane[2][lyidx][lxidx] = cur_2_1;
		__syncthreads();

		temp_k = nz-5;
		if (threadIdx.x > 0 && threadIdx.x < blockDim.x-1 &&
			threadIdx.y > 0 && threadIdx.y < blockDim.y-1){
			if (global_i > 0 && global_i < nx-1 &&
				global_j > 0 && global_j < ny-1){
				cur_1_3 = cur_0_2;
				stencil_only(2,3,1)
				write_global_cal(gidx,slice,temp_k)
			}else{
				cur_1_3 = cur_0_1;	
				cur_2_3 = cur_1_1;
			}
		}

		__syncthreads();
		//cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		//cur_plane[0][lyidx][lxidx] = cur_0_1;
		cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
		cur_plane[1][lyidx][lxidx] = cur_1_1;
		cur_2_0 = cur_2_1; cur_2_1 = cur_2_2; cur_2_2 = cur_2_3;
		cur_plane[2][lyidx][lxidx] = cur_2_1;
		__syncthreads();

		temp_k = nz-4;
		if (threadIdx.x > 0 && threadIdx.x < blockDim.x-1 &&
			threadIdx.y > 0 && threadIdx.y < blockDim.y-1){
			if (global_i > 0 && global_i < nx-1 &&
				global_j > 0 && global_j < ny-1){
				stencil_only(2,3,1)
				write_global_cal(gidx,slice,temp_k)
			}else{
				cur_1_3 = cur_0_1;	
				cur_2_3 = cur_1_1;
			}
		}

		__syncthreads();
		//cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		//cur_plane[0][lyidx][lxidx] = cur_0_1;
		//cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
		//cur_plane[1][lyidx][lxidx] = cur_1_1;
		cur_2_0 = cur_2_1; cur_2_1 = cur_2_2; cur_2_2 = cur_2_3;
		cur_plane[2][lyidx][lxidx] = cur_2_1;
		__syncthreads();

		temp_k = nz-3;
		if (global_i > 0 && global_i < nx-1 &&
			global_j > 0 && global_j < ny-1){
			cur_2_3 = cur_1_2;
			write_global_cal(gidx,slice,temp_k)
		}else{
			cur_1_3 = cur_0_1;	
			cur_2_3 = cur_1_1;
		}

		__syncthreads();
		//cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		//cur_plane[0][lyidx][lxidx] = cur_0_1;
		//cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
		//cur_plane[1][lyidx][lxidx] = cur_1_1;
		cur_2_0 = cur_2_1; cur_2_1 = cur_2_2; cur_2_2 = cur_2_3;
		cur_plane[2][lyidx][lxidx] = cur_2_1;
		__syncthreads();

		temp_k = nz-2;
		if (global_i > 0 && global_i < nx-1 &&
			global_j > 0 && global_j < ny-1){
			write_global_cal(gidx, slice, temp_k)
		}

		//__syncthreads();
		//cur_0_0 = cur_0_1; cur_0_1 = cur_0_2; cur_0_2 = cur_0_3;
		//cur_plane[0][lyidx][lxidx] = cur_0_1;
		//cur_1_0 = cur_1_1; cur_1_1 = cur_1_2; cur_1_2 = cur_1_3;
		//cur_plane[1][lyidx][lxidx] = cur_1_1;
		//cur_2_0 = cur_2_1; cur_2_1 = cur_2_2; cur_2_2 = cur_2_3;
		//cur_plane[2][lyidx][lxidx] = cur_2_1;
		//__syncthreads();

		temp_k = nz-1;
		if (global_i > 0 && global_i < nx-1 &&
			global_j > 0 && global_j < ny-1){
			write_global_copy(gidx,slice,temp_k)
		}
	}
	return;
}

//#define check
#define checkT dimT

int main(){

	int64_t size = sizeof(REAL)*NX*NY*NZ;
	REAL* host_A = (REAL*)malloc(size);
	REAL* host_B = (REAL*)malloc(size);
	REAL* host_RES = (REAL*)malloc(size);

	for (int64_t k = 0; k < NZ; k++)
		for (int64_t j = 0; j < NY; j++)
			for (int64_t i = 0; i < NX; i++){
				host_B[k*NY*NX+j*NX+i] = i - j + 1.0/(k+1);	
				host_A[k*NY*NX+j*NX+i] = i - j + 1.0/(k+1);	
			}

	//cudaSetDevice(2);
	REAL *dev_A, *dev_B;
	cudaMalloc(&dev_A, size);
	cudaMalloc(&dev_B, size);
	cudaMemcpy(dev_B, host_B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_A, host_B, size, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float elapsed_time;
	double flops;

	//dim3 threadPerBlock(BX, BY, BZ);
	//dim3 blockPerGrid((NX+BX-1)/BX, (NY+BY-1)/BY, GZ);

	///////////////////////////////////////////////////////////////
	//baseline
	cudaEventRecord(start, 0);
	/*
	for (int64_t t = 0; t < T; t++){
		baseline<<<blockPerGrid, threadPerBlock>>>(dev_A, dev_B, NX, NY, NZ);		
		REAL* tmp = dev_B;
		dev_B = dev_A;
		dev_A = tmp;
	}
	*/
	dim3 tpb(BX, BY, BZ);
	dim3 bpg((NX+BX-2*dimT-1)/(BX-2*dimT), 
			 (NY+BY-2*dimT-1)/(BY-2*dimT), 1);
	for (int64_t t = 0; t < T; t += dimT){
		temporal_blocking<<<bpg, tpb>>>(dev_A, dev_B, NX, NY, NZ);
		REAL* tmp = dev_B;
		dev_B = dev_A;
		dev_A = tmp;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaError_t err;
	if ((err=cudaGetLastError()) != cudaSuccess)
		printf("baseline: wrong: %s!!!\n", cudaGetErrorString(err));
	cudaEventElapsedTime(&elapsed_time, start, stop);

	printf("baseline: elapsed time = %f ms\n", elapsed_time);
	flops = 1.0*13*(NX-2)*(NY-2)*(NZ-2)*T/1.e+6;
	//flops = 1.0*13*NX*NY*NZ*T/1.e+6;
	flops /= elapsed_time;
	printf("baseline: Gflops = %lf\n", flops);
	///////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////
	//check result
#ifdef check
	cudaMemcpy(host_RES, dev_B, size, cudaMemcpyDeviceToHost);
	for (int64_t t = 0; t < T; t++){
		for (int64_t k = 1; k < NZ-1; k++)
			for (int64_t j = 1; j < NY-1; j++)
				for (int64_t i = 1; i < NX-1; i++)
					host_A[k*NY*NX+j*NX+i] = 
						ce*host_B[k*NY*NX+j*NX+i+1]
					   +cw*host_B[k*NY*NX+j*NX+i-1]
					   +cs*host_B[k*NY*NX+(j+1)*NX+i]
					   +cn*host_B[k*NY*NX+(j-1)*NX+i]
					   +ct*host_B[(k+1)*NY*NX+j*NX+i]
					   +cb*host_B[(k-1)*NY*NX+j*NX+i]
					   +cc*host_B[k*NY*NX+j*NX+i];

		REAL *tmp = host_A;
		host_A = host_B;
		host_B = tmp;
	}
	for (int64_t k = 0; k < NZ; k++)
		for (int64_t j = 0; j < NY; j++)
			for (int64_t i = 0; i < NX; i++)
				if (host_B[k*NY*NX+j*NX+i] != 
						host_RES[k*NY*NX+j*NX+i])
					printf("host_B[%d][%d][%d] = %f\t" 
						   "host_RES[%d][%d][%d] = %f\n", 
						   k, j, i, host_B[k*NY*NX+j*NX+i], 
						   k, j, i, host_RES[k*NY*NX+j*NX+i]);
#endif


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return 0;
}
