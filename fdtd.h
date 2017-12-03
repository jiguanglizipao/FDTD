#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <iostream>

#define CUDA_ASSERT(expr) { \
    cudaError_t t = (expr); \
    if (t != cudaSuccess) { \
        fprintf(stderr, "ERROR AT [ %s ] : %s\n", #expr, cudaGetErrorString(t)); \
        assert(false); \
    } \
}
//template <typename Function, Function f, typename... Arguments>
//__global__ void kernel(Arguments... args)
//{
//    printf("value = %d\n", f(args...)); 
//}
//
//__device__ __forceinline__ int function1(int x, int y)
//{
//    for(int i=0;i<y;i++)x=x*x;
//    return x;
//}
//
//__device__ __forceinline__ int function2(int x, int y)
//{
//    return x+y;
//}
//
//class class1
//{
//public:
//    __device__ __forceinline__ int operator() (int x, int y) const
//    {
//        for(int i=0;i<y;i++)x=x*x;
//        return x;
//    }
//};
//
//class class2
//{
//public:
//    __device__ __forceinline__ int operator() (int x, int y) const
//    {
//        return x+y;
//    }
//};
//
//template <typename Function, typename... Arguments>
//__global__ void kernel(Function f, Arguments... args)
//{
//    printf("value = %d\n", f(args...)); 
//}
//
//__global__ void raw1(int x, int y)
//{
//    printf("value = %d\n", function1(x, y)); 
//}
//
//__global__ void raw2(int x, int y)
//{
//    printf("value = %d\n", function2(x, y)); 
//}

template <typename DataType, size_t M, size_t blocks, typename Function, typename... Arguments>
__launch_bounds__(1024, blocks)
__global__ static void prop_center_kernel(DataType *p0, DataType *p1, size_t _nx, int _ny, int _nz, int offz, Function kernel, Arguments... args)
{
    const size_t PADDINGL=M, PADDINGR=(_nx%32>(32-M))?(64-M-_nx%32):(32-M-_nx%32);
    const size_t BLOCK_SIZE = 32;
    __shared__ DataType p1s[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M];
    __shared__ DataType p1st[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M];

    size_t ig = (blockIdx.x * blockDim.x + threadIdx.x + M + PADDINGL);
    size_t jg = (blockIdx.y * blockDim.y + threadIdx.y + M);

    size_t il = threadIdx.x + M;
    size_t jl = threadIdx.y + M; 

    DataType p1z[2*M+1];
    size_t _n12 = (_nx+PADDINGL+PADDINGR)*_ny;
    size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg;
    int64_t addr_fwd = addr-M*_n12-_n12;

    int64_t ir, jr, offr;
    if(threadIdx.y < M){
        ir = il;
        jr = jl - M;
        offr = -M * (_nx+PADDINGL+PADDINGR);
    }
    else if(threadIdx.y < 2*M){
        ir = il;
        jr = jl + (BLOCK_SIZE-M);
        offr = (BLOCK_SIZE-M) * (_nx+PADDINGL+PADDINGR);
    }
    else if(threadIdx.y < 3*M){
        ir = jl - 3*M;
        jr = il;
        offr = (il - jl) * (_nx+PADDINGL+PADDINGR) + (jl - 3*M - il);
    }
    else if(threadIdx.y < 4*M){
        ir = jl + (BLOCK_SIZE-3*M);
        jr = il;
        offr = (il - jl) * (_nx+PADDINGL+PADDINGR) + (jl + (BLOCK_SIZE-3*M) - il);
    }
    if(threadIdx.y < 4*M)
    {
        p1st[ir][jr] = p1s[jr][ir] = p1[addr + offr];
    }
    __syncthreads();

    #pragma unroll 
    for(size_t t=1;t<M;t++) p1z[t] = p1[addr_fwd+=_n12];
    p1z[M] = p1st[il][jl] = p1s[jl][il] = p1[addr_fwd+=_n12];
    #pragma unroll 
    for(size_t t=M+1;t<=2*M;t++) p1z[t]=p1[addr_fwd+=_n12];

//    #pragma unroll 2
    for(size_t yl=0; yl<_nz; yl++)
    {
        #pragma unroll 
        for(size_t t=0;t<M-1;t++) p1z[t] = p1z[t+1];
        p1z[M-1] = p1s[jl][il];
        p1z[M] = p1st[il][jl] = p1s[jl][il] = p1z[M+1];
        #pragma unroll 
        for(size_t t=M+1;t<2*M;t++) p1z[t] = p1z[t+1];
        p1z[2*M] = p1[addr_fwd+=_n12];

        __syncthreads();
        kernel(yl+offz, jg, ig-PADDINGL, p0[addr], p1z+M, &p1st[il][jl], &p1s[jl][il], args...);
        __syncthreads();
        addr+=_n12;
        if(threadIdx.y < 4*M)
            p1st[ir][jr] = p1s[jr][ir] = p1[addr + offr];
    }
}

template <typename DataType, size_t M, size_t blocks, typename Function, typename... Arguments>
__launch_bounds__(1024, blocks)
__global__ static void prop_halo_kernel(DataType *p0, DataType *p1, int _nx, int _ny, int _nz, int nx2, int ny2, int offx, int offy, int offz, Function kernel, Arguments... args)
{
    const size_t PADDINGL=M, PADDINGR=(_nx%32>(32-M))?(64-M-_nx%32):(32-M-_nx%32);
    const size_t BLOCK_SIZE = 32;
    __shared__ DataType p1s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DataType p1st[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M];

    size_t ig = ((blockIdx.x+offx) * (blockDim.x-2*M) + threadIdx.x + PADDINGL);
    size_t jg = ((blockIdx.y+offy) * (blockDim.y-2*M) + threadIdx.y);
    if(ig < nx2+PADDINGL && jg < ny2)return;
    if(ig >= _nx+PADDINGL || jg >= _ny)return;

    bool flag = (threadIdx.x>=M && threadIdx.x<blockDim.x-M && threadIdx.y>=M && threadIdx.y<blockDim.y-M 
        && ig>=M+PADDINGL && ig<_nx-M+PADDINGL && jg>=M && jg<_ny-M
        && (ig >= nx2+M+PADDINGL || jg >= ny2+M));

    size_t il = threadIdx.x;
    size_t jl = threadIdx.y; 

    DataType p1z[2*M+1];
    size_t _n12 = (_nx+PADDINGL+PADDINGR)*_ny;
    size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg;
    int64_t addr_fwd = addr-M*_n12-_n12;

    #pragma unroll 
    for(size_t t=1;t<M;t++) p1z[t] = p1[addr_fwd+=_n12];
    p1z[M] = p1st[il][jl] = p1s[jl][il] = p1[addr_fwd+=_n12];
    #pragma unroll 
    for(size_t t=M+1;t<=2*M;t++) p1z[t]=p1[addr_fwd+=_n12];

//    #pragma unroll 9
    for(size_t yl=0; yl<_nz; yl++)
    {
        #pragma unroll 
        for(size_t t=0;t<M-1;t++) p1z[t] = p1z[t+1];
        p1z[M-1] = p1s[jl][il];
        p1z[M] = p1st[il][jl] = p1s[jl][il] = p1z[M+1];
        #pragma unroll 
        for(size_t t=M+1;t<2*M;t++) p1z[t] = p1z[t+1];
        p1z[2*M] = p1[addr_fwd+=_n12];

        __syncthreads();
        if(flag)
            kernel(yl+offz, jg, ig-PADDINGL, p0[addr], p1z+M, &p1st[il][jl], &p1s[jl][il], args...);
        __syncthreads();
        addr+=_n12;
    }
}

template <typename DataType, size_t ZSize, size_t YSize, size_t XSize, size_t M>
class FDTD
{
private:
    std::map<std::string, std::shared_ptr<DataType>> memoryPool;
    const size_t PADDINGL, PADDINGR;
    const size_t BLOCK_SIZE = 32, THREADS = 256;
public:
    __host__ FDTD()
        : PADDINGL(M), PADDINGR((XSize%32>(32-M))?(64-M-XSize%32):(32-M-XSize%32))
    {
        assert(M <= 8 && ZSize >= 2*M+1);
    }

    template <typename T=DataType>
    __host__ T* malloc(std::string name, size_t length)
    {
        assert(memoryPool.find(name) == memoryPool.end());
        T* p = nullptr;
        CUDA_ASSERT(cudaMalloc(&p, sizeof(T)*length));
        memoryPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
        return p;
    }

    __host__ void free(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        memoryPool.erase(memoryPool.find(name));
    }

    template <typename T>
    __host__ void transferToGPU(std::string name, T* cpu, size_t length)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        CUDA_ASSERT(cudaMemcpy(memoryPool[name].get(), cpu, sizeof(T)*length, cudaMemcpyHostToDevice));
    }

    template <typename T>
    __host__ void transferToCPU(T* cpu, std::string name, size_t length)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        CUDA_ASSERT(cudaMemcpy(cpu, memoryPool[name].get(), sizeof(T)*length, cudaMemcpyDeviceToHost));
    }

    __host__ DataType* mallocCube(std::string name)
    {
        assert(memoryPool.find(name) == memoryPool.end());
        DataType* p = nullptr;
        CUDA_ASSERT(cudaMalloc(&p, sizeof(DataType)*ZSize*YSize*(PADDINGL+XSize+PADDINGR)));
        memoryPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
        return p;
    }

    size_t GET(size_t z, size_t y, size_t x, size_t nz, size_t ny, size_t nx)
    {
        return z*(ny*nx)+y*nx+x;
    }

    __host__ void transferCubeToGPU(std::string name, DataType* cpu)
    {
        DataType *tmp = (DataType*)aligned_alloc(64, sizeof(DataType)*ZSize*YSize*(PADDINGL+XSize+PADDINGR));
        memset(tmp, 0, sizeof(DataType)*ZSize*YSize*(PADDINGL+XSize+PADDINGR));
        for(size_t i=0;i<ZSize;i++)
            for(size_t j=0;j<YSize;j++)
                for(size_t k=0;k<XSize;k++)
                    tmp[GET(i,j,k+PADDINGL,ZSize,YSize,PADDINGL+XSize+PADDINGR)] = cpu[GET(i,j,k,ZSize,YSize,XSize)];
        CUDA_ASSERT(cudaMemcpy(memoryPool[name].get(), tmp, sizeof(DataType)*ZSize*YSize*(PADDINGL+XSize+PADDINGR), cudaMemcpyHostToDevice));
        ::free(tmp);
    }

    __host__ void transferCubeToCPU(DataType* cpu, std::string name)
    {
        DataType *tmp = (DataType*)aligned_alloc(64, sizeof(DataType)*ZSize*YSize*(PADDINGL+XSize+PADDINGR));
        CUDA_ASSERT(cudaMemcpy(tmp, memoryPool[name].get(), sizeof(DataType)*ZSize*YSize*(PADDINGL+XSize+PADDINGR), cudaMemcpyDeviceToHost));
        for(size_t i=0;i<ZSize;i++)
            for(size_t j=0;j<YSize;j++)
                for(size_t k=0;k<XSize;k++)
                    cpu[GET(i,j,k,ZSize,YSize,XSize)] = tmp[GET(i,j,k+PADDINGL,ZSize,YSize,PADDINGL+XSize+PADDINGR)];
        ::free(tmp);
    }

    template <typename Function, typename... Arguments>
    __host__ void propagate(std::string output, std::string input, bool spill, Function kernel, Arguments... args)
    {
        assert(memoryPool.find(input) != memoryPool.end());
        assert(memoryPool.find(output) != memoryPool.end());
        DataType *p0 = memoryPool[output].get(), *p1 = memoryPool[input].get();
        size_t _nx = XSize, _ny = YSize, _nz = ZSize;
        size_t nz = _nz-2*M;
        size_t nx2 = (_nx - 2*M) / BLOCK_SIZE * BLOCK_SIZE + 2*M;
        size_t ny2 = (_ny - 2*M) / BLOCK_SIZE * BLOCK_SIZE + 2*M;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        {
            dim3 grid((nx2-2*M)/BLOCK_SIZE, (ny2-2*M)/BLOCK_SIZE);
            if(spill)
                prop_center_kernel<DataType, M, 2, Function, Arguments...><<<grid, block>>>(p0+M*_ny*(_nx+PADDINGL+PADDINGR), p1+M*_ny*(_nx+PADDINGL+PADDINGR), _nx, _ny, nz, M, kernel, args...);
            else
                prop_center_kernel<DataType, M, 1, Function, Arguments...><<<grid, block>>>(p0+M*_ny*(_nx+PADDINGL+PADDINGR), p1+M*_ny*(_nx+PADDINGL+PADDINGR), _nx, _ny, nz, M, kernel, args...);
        }
        {
            size_t xs = ((nx2-2*M))/(BLOCK_SIZE-2*M);
            size_t ys = ((ny2-2*M))/(BLOCK_SIZE-2*M);
            size_t xt = ((_nx-2*M)+(BLOCK_SIZE-2*M)-1)/(BLOCK_SIZE-2*M);
            size_t yt = ((_ny-2*M)+(BLOCK_SIZE-2*M)-1)/(BLOCK_SIZE-2*M);
            {
                dim3 grid(xt-xs, yt);
                if(spill)
                    prop_halo_kernel<DataType, M, 2, Function, Arguments...><<<grid, block>>>(p0+M*_ny*(_nx+PADDINGL+PADDINGR), p1+M*_ny*(_nx+PADDINGL+PADDINGR),  _nx, _ny, nz, nx2-2*M, ny2-2*M, xs, 0, M, kernel, args...);
                else
                    prop_halo_kernel<DataType, M, 1, Function, Arguments...><<<grid, block>>>(p0+M*_ny*(_nx+PADDINGL+PADDINGR), p1+M*_ny*(_nx+PADDINGL+PADDINGR),  _nx, _ny, nz, nx2-2*M, ny2-2*M, xs, 0, M, kernel, args...);
            }
            {
                dim3 grid(xs, yt-ys);
                if(spill)
                    prop_halo_kernel<DataType, M, 2, Function, Arguments...><<<grid, block>>>(p0+M*_ny*(_nx+PADDINGL+PADDINGR), p1+M*_ny*(_nx+PADDINGL+PADDINGR), _nx, _ny, nz, nx2-2*M, ny2-2*M, 0, ys, M, kernel, args...);
                else
                    prop_halo_kernel<DataType, M, 1, Function, Arguments...><<<grid, block>>>(p0+M*_ny*(_nx+PADDINGL+PADDINGR), p1+M*_ny*(_nx+PADDINGL+PADDINGR), _nx, _ny, nz, nx2-2*M, ny2-2*M, 0, ys, M, kernel, args...);
            }
        }
        CUDA_ASSERT(cudaDeviceSynchronize());
    }

};


