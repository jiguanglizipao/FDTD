#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
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

template <typename DataType, typename gpu_size_t, typename gpu_signed_size_t, gpu_size_t M, gpu_size_t blocks, gpu_size_t PADDINGL, gpu_size_t PADDINGR, gpu_size_t BLOCK_SIZE, typename Function, typename... Arguments>
__launch_bounds__(512, blocks)
__global__ static void prop_center_nosm_kernel(DataType *p0, DataType *p1, gpu_size_t _nx, gpu_size_t _ny, gpu_size_t _nz, gpu_size_t offz, gpu_size_t offaddr, Function kernel, Arguments... args)
{
    gpu_size_t ig = (blockIdx.x * blockDim.x + threadIdx.x + PADDINGL);
    gpu_size_t jg = (blockIdx.y * blockDim.y + threadIdx.y );
    if(ig >= _nx+PADDINGL-M || jg >= _ny-M)return;
    if(ig < M+PADDINGL || jg < M)return;

    gpu_size_t _n12 = (_nx+PADDINGL+PADDINGR)*_ny;
    gpu_size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg+offaddr;

    #pragma unroll 2
    for(gpu_size_t yl=0; yl<_nz; yl++)
    {
        kernel(yl+offz, jg, ig-PADDINGL, addr, p0, &p1[addr], _n12, &p1[addr], _nx+PADDINGL+PADDINGR, &p1[addr], 1, args...);
        addr+=_n12;
    }
}

template <typename DataType, typename gpu_size_t, typename gpu_signed_size_t, gpu_size_t M, gpu_size_t blocks, gpu_size_t PADDINGL, gpu_size_t PADDINGR, gpu_size_t BLOCK_SIZE, typename Function, typename... Arguments>
__launch_bounds__(1024, blocks)
__global__ static void prop_center_kernel(DataType *p0, DataType *p1, gpu_size_t _nx, gpu_size_t _ny, gpu_size_t _nz, gpu_size_t offz, gpu_size_t offaddr, Function kernel, Arguments... args)
{
    __shared__ DataType p1s[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M+1];

    gpu_size_t ig = (blockIdx.x * blockDim.x + threadIdx.x + M + PADDINGL);
    gpu_size_t jg = (blockIdx.y * blockDim.y + threadIdx.y + M);

    gpu_size_t il = threadIdx.x + M;
    gpu_size_t jl = threadIdx.y + M; 

    DataType p1z[2*M+1];
    gpu_size_t _n12 = (_nx+PADDINGL+PADDINGR)*_ny;
    gpu_size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg+offaddr;
    gpu_signed_size_t addr_fwd = addr-M*_n12-_n12;

    gpu_signed_size_t ir, jr, offr;
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
        p1s[jr][ir] = p1[addr + offr];
    }
    __syncthreads();

    #pragma unroll 
    for(gpu_size_t t=1;t<M;t++) p1z[t] = p1[addr_fwd+=_n12];
    p1z[M] = p1s[jl][il] = p1[addr_fwd+=_n12];
    #pragma unroll 
    for(gpu_size_t t=M+1;t<=2*M;t++) p1z[t]=p1[addr_fwd+=_n12];

//    #pragma unroll 2
    for(gpu_size_t yl=0; yl<_nz; yl++)
    {
        #pragma unroll 
        for(gpu_size_t t=0;t<M-1;t++) p1z[t] = p1z[t+1];
        p1z[M-1] = p1s[jl][il];
        p1z[M] = p1s[jl][il] = p1z[M+1];
        #pragma unroll 
        for(gpu_size_t t=M+1;t<2*M;t++) p1z[t] = p1z[t+1];
        p1z[2*M] = p1[addr_fwd+=_n12];

        __syncthreads();
        kernel(yl+offz, jg, ig-PADDINGL, addr, p0, p1z+M, 1, &p1s[jl][il], BLOCK_SIZE+2*M+1, &p1s[jl][il], 1, args...);
        __syncthreads();
        addr+=_n12;
        if(threadIdx.y < 4*M)
            p1s[jr][ir] = p1[addr + offr];
    }
}

template <typename DataType, typename gpu_size_t, typename gpu_signed_size_t, gpu_size_t M, gpu_size_t blocks, gpu_size_t PADDINGL, gpu_size_t PADDINGR, gpu_size_t BLOCK_SIZE, typename Function, typename... Arguments>
__launch_bounds__(1024, blocks)
__global__ static void prop_halo_kernel(DataType *p0, DataType *p1, gpu_size_t _nx, gpu_size_t _ny, gpu_size_t _nz, gpu_size_t nx2, gpu_size_t ny2, gpu_size_t offx, gpu_size_t offy, gpu_size_t offz, gpu_size_t offaddr, Function kernel, Arguments... args)
{
    __shared__ DataType p1s[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M+1];

    gpu_size_t ig = ((blockIdx.x+offx) * (blockDim.x-2*M) + threadIdx.x + PADDINGL);
    gpu_size_t jg = ((blockIdx.y+offy) * (blockDim.y-2*M) + threadIdx.y);
    if(ig < nx2+PADDINGL && jg < ny2)return;
    if(ig >= _nx+PADDINGL || jg >= _ny)return;

    bool flag = (threadIdx.x>=M && threadIdx.x<blockDim.x-M && threadIdx.y>=M && threadIdx.y<blockDim.y-M 
        && ig>=M+PADDINGL && ig<_nx-M+PADDINGL && jg>=M && jg<_ny-M
        && (ig >= nx2+M+PADDINGL || jg >= ny2+M));

    gpu_size_t il = threadIdx.x;
    gpu_size_t jl = threadIdx.y; 

    DataType p1z[2*M+1];
    gpu_size_t _n12 = (_nx+PADDINGL+PADDINGR)*_ny;
    gpu_size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg+offaddr;
    gpu_signed_size_t addr_fwd = addr-M*_n12-_n12;

    #pragma unroll 
    for(gpu_size_t t=1;t<M;t++) p1z[t] = p1[addr_fwd+=_n12];
    p1z[M] = p1s[jl][il] = p1[addr_fwd+=_n12];
    #pragma unroll 
    for(gpu_size_t t=M+1;t<=2*M;t++) p1z[t]=p1[addr_fwd+=_n12];

//    #pragma unroll 9
    for(gpu_size_t yl=0; yl<_nz; yl++)
    {
        #pragma unroll 
        for(gpu_size_t t=0;t<M-1;t++) p1z[t] = p1z[t+1];
        p1z[M-1] = p1s[jl][il];
        p1z[M] = p1s[jl][il] = p1z[M+1];
        #pragma unroll 
        for(gpu_size_t t=M+1;t<2*M;t++) p1z[t] = p1z[t+1];
        p1z[2*M] = p1[addr_fwd+=_n12];

        __syncthreads();
        if(flag)
            kernel(yl+offz, jg, ig-PADDINGL, addr, p0, p1z+M, 1, &p1s[jl][il], BLOCK_SIZE+2*M+1, &p1s[jl][il], 1, args...);
        __syncthreads();
        addr+=_n12;
    }
}

template <typename DataType, typename gpu_size_t, typename gpu_signed_size_t, gpu_size_t M, gpu_size_t blocks, gpu_size_t PADDINGL, gpu_size_t PADDINGR, gpu_size_t BLOCK_SIZE, typename Function, typename... Arguments>
__launch_bounds__(1024, blocks)
__global__ static void prop_small_kernel(DataType *p0, DataType *p1, gpu_size_t _nx, gpu_size_t _ny, gpu_size_t _nz, gpu_size_t offz, gpu_size_t offaddr, Function kernel, Arguments... args)
{
    __shared__ DataType p1s[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M+1];

    gpu_size_t ig = (blockIdx.x * (blockDim.x-2*M) + threadIdx.x + PADDINGL);
    gpu_size_t jg = (blockIdx.y * (blockDim.y-2*M) + threadIdx.y);
    if(ig >= _nx+PADDINGL || jg >= _ny)return;

    bool flag = (threadIdx.x>=M && threadIdx.x<blockDim.x-M && threadIdx.y>=M && threadIdx.y<blockDim.y-M 
        && ig>=M+PADDINGL && ig<_nx-M+PADDINGL && jg>=M && jg<_ny-M);

    gpu_size_t il = threadIdx.x;
    gpu_size_t jl = threadIdx.y; 

    DataType p1z[2*M+1];
    gpu_size_t _n12 = (_nx+PADDINGL+PADDINGR)*_ny;
    gpu_size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg+offaddr;
    gpu_signed_size_t addr_fwd = addr-M*_n12-_n12;

    #pragma unroll 
    for(gpu_size_t t=1;t<M;t++) p1z[t] = p1[addr_fwd+=_n12];
    p1z[M] = p1s[jl][il] = p1[addr_fwd+=_n12];
    #pragma unroll 
    for(gpu_size_t t=M+1;t<=2*M;t++) p1z[t]=p1[addr_fwd+=_n12];

//    #pragma unroll 9
    for(gpu_size_t yl=0; yl<_nz; yl++)
    {
        #pragma unroll 
        for(gpu_size_t t=0;t<M-1;t++) p1z[t] = p1z[t+1];
        p1z[M-1] = p1s[jl][il];
        p1z[M] = p1s[jl][il] = p1z[M+1];
        #pragma unroll 
        for(gpu_size_t t=M+1;t<2*M;t++) p1z[t] = p1z[t+1];
        p1z[2*M] = p1[addr_fwd+=_n12];

        __syncthreads();
        if(flag)
            kernel(yl+offz, jg, ig-PADDINGL, addr, p0, p1z+M, 1, &p1s[jl][il], BLOCK_SIZE+2*M+1, &p1s[jl][il], 1, args...);
        __syncthreads();
        addr+=_n12;
    }
}

template <typename gpu_size_t, typename gpu_signed_size_t, typename Function, typename... Arguments>
 __global__ void inject_kernel(gpu_size_t length, Function kernel, Arguments... args) 
{
    gpu_size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= length) return;
    kernel(i, args...);
}

template <typename gpu_size_t, typename gpu_signed_size_t, typename Function, typename... Arguments>
 __global__ void filt_kernel(gpu_size_t _nx, gpu_size_t _ny, gpu_size_t zbase, gpu_size_t zoff, gpu_size_t PADDINGL, gpu_size_t PADDINGR, Function kernel, Arguments... args) 
{
    gpu_size_t x = blockDim.x*blockIdx.x+threadIdx.x;
    if(x >= _nx)return;
    gpu_size_t y = blockIdx.y;
    gpu_size_t z = blockIdx.z;
    gpu_size_t addr = (z+zoff)*(_ny*(_nx+PADDINGL+PADDINGR))+y*(_nx+PADDINGL+PADDINGR)+x+PADDINGL;

    kernel(zbase+z, y, x, addr, args...);
}

template <typename DataType, size_t ZSize, size_t YSize, size_t XSize, size_t M, typename gpu_size_t=uint32_t, typename gpu_signed_size_t=int32_t, size_t BLOCK_SIZE=32, size_t THREADS=256, size_t PADDINGL=(M==1)?0:M, size_t PADDINGR=(M==1)?(BLOCK_SIZE-XSize%BLOCK_SIZE):((XSize%BLOCK_SIZE>(BLOCK_SIZE-M))?(2*BLOCK_SIZE-M-XSize%BLOCK_SIZE):(BLOCK_SIZE-M-XSize%BLOCK_SIZE))>
class Stencil
{
private:
    std::map<std::string, std::shared_ptr<DataType>> memoryPool;
    std::map<std::string, std::shared_ptr<DataType>> topPool;
    std::map<std::string, std::shared_ptr<DataType>> buttomPool;
    std::map<std::string, size_t> addrLen, topAddrLen, buttomAddrLen;
    size_t nprocs, rank, size2, off2, off, size, PXSize;
    std::vector<size_t> mpiSize, mpiOff;
    MPI::Datatype mpi_datatype;

    size_t GET(size_t z, size_t y, size_t x, size_t nz, size_t ny, size_t nx)
    {
        return z*(ny*nx)+y*nx+x;
    }

    __host__ void MPI_COMM_WORLD_BCAST(DataType* data, size_t size, int root)
    {
        const size_t MAXINT = 1024*1024*512;
        size_t off=0;
        for(;size>MAXINT;size-=MAXINT,off+=MAXINT)
            MPI::COMM_WORLD.Bcast(data+off, MAXINT, mpi_datatype, root);
        MPI::COMM_WORLD.Bcast(data+off, size, mpi_datatype, root);
    }

public:
   __host__ Stencil()
    {
        assert(M <= 8 && ZSize >= 2*M+1);
        memoryPool.clear();
        if(!MPI::Is_initialized()) MPI::Init();
        nprocs = MPI::COMM_WORLD.Get_size();
        rank = MPI::COMM_WORLD.Get_rank();
        mpi_datatype = MPI::Datatype::Match_size(MPI_TYPECLASS_REAL, sizeof(DataType));

        mpiSize.resize(nprocs);
        mpiOff.resize(nprocs+1);
        for(int i=0;i<nprocs;i++)
        {
            mpiSize[i] = ZSize/nprocs+(ZSize%nprocs>i);
            assert(mpiSize[i] > 2*M);
            mpiOff[i] = (i==0)?0:(mpiOff[i-1]+mpiSize[i-1]);
        }
        mpiOff[nprocs] = ZSize;

        PXSize = PADDINGL+XSize+PADDINGR;
        size2 = size_t(mpiSize[rank]+(2*M-((rank==0)?M:0)-((rank==nprocs-1)?M:0))) * YSize * PXSize;
        size = size_t(mpiSize[rank]) * YSize * PXSize;
        off2 = size_t(mpiOff[rank]-((rank == 0)?0:M)) * YSize * PXSize;
        off = size_t(mpiOff[rank]) * YSize * PXSize;
    }

    __host__ ~Stencil()
    {
        MPI::Finalize();
    }

    __host__ size_t getRank()
    {
        return rank;
    }

    __host__ size_t getProcs()
    {
        return nprocs;
    }

    __host__ void sync()
    {
        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    __host__ void barrier()
    {
        MPI::COMM_WORLD.Barrier();
    }

    template <typename T=DataType>
    __host__ T* malloc(std::string name, size_t length, bool is_address = false)
    {
         assert(memoryPool.find(name) == memoryPool.end());
        T* p = nullptr;
        CUDA_ASSERT(cudaMalloc(&p, sizeof(T)*length));
        memoryPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
        
        if(is_address)
        {
            addrLen[name] = 0;
            if(rank != 0)
            {
                CUDA_ASSERT(cudaMalloc(&p, sizeof(T)*length));
                topPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
                topAddrLen[name] = 0;
            }
            if(rank != nprocs-1)
            {
                CUDA_ASSERT(cudaMalloc(&p, sizeof(T)*length));
                buttomPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
                buttomAddrLen[name] = 0;
            }
        }

        return p;
    }

    __host__ void free(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        if(memoryPool.find(name) != memoryPool.end()) memoryPool.erase(memoryPool.find(name));
        if(topPool.find(name) != topPool.end()) topPool.erase(topPool.find(name));
        if(buttomPool.find(name) != buttomPool.end()) buttomPool.erase(buttomPool.find(name));
        if(addrLen.find(name) != addrLen.end()) addrLen.erase(addrLen.find(name));
        if(topAddrLen.find(name) != topAddrLen.end()) topAddrLen.erase(topAddrLen.find(name));
        if(buttomAddrLen.find(name) != buttomAddrLen.end()) buttomAddrLen.erase(buttomAddrLen.find(name));
    }

    template <typename T=DataType>
    __host__ T* get(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        return (T*)memoryPool[name].get();
    }

    template <typename T=DataType>
    __host__ T* getHaloTop(std::string name)
    {
        if(topPool.find(name) == topPool.end()) return nullptr;
        return (T*)topPool[name].get();
    }

    template <typename T=DataType>
    __host__ T* getHaloButtom(std::string name)
    {
        if(buttomPool.find(name) == buttomPool.end()) return nullptr;
        return (T*)buttomPool[name].get();
    }

    size_t getAddrSize(std::string name)
    {
        if(addrLen.find(name) == addrLen.end()) return 0;
        return addrLen[name];
    }

    size_t getAddrHaloTopSize(std::string name)
    {
        if(topAddrLen.find(name) == topAddrLen.end()) return 0;
        return topAddrLen[name];
    }
    
    size_t getAddrHaloButtomSize(std::string name)
    {
        if(buttomAddrLen.find(name) == buttomAddrLen.end()) return 0;
        return buttomAddrLen[name];
    }

    template <typename T>
    __host__ void transferToGPU(std::string name, T* cpu, size_t length)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        if(addrLen.find(name) == addrLen.end())
        {
            CUDA_ASSERT(cudaMemcpy(memoryPool[name].get(), cpu, sizeof(T)*length, cudaMemcpyHostToDevice));
        }
        else
        {
            std::vector<T> addr, top, buttom;
            for(size_t i=0;i<length;i++)
            {
                size_t z = cpu[i]/(YSize*XSize), y = (cpu[i]%(YSize*XSize))/XSize, x = (cpu[i]%(YSize*XSize))%XSize;
                if(z < mpiOff[rank] || z >= mpiOff[rank+1]) continue;
                addr.push_back((z-(mpiOff[rank]-((rank == 0)?0:M)))*(YSize*PXSize)+y*PXSize+(x+PADDINGL));
                if(rank != 0 && z < mpiOff[rank]+2*M) top.push_back(addr.back());
                if(rank != nprocs-1 && z >= mpiOff[rank+1]-2*M) buttom.push_back((z-mpiOff[rank+1]+2*M)*(YSize*PXSize)+y*PXSize+(x+PADDINGL));
            }
            if(!addr.empty()) CUDA_ASSERT(cudaMemcpy(memoryPool[name].get(), addr.data(), sizeof(T)*addr.size(), cudaMemcpyHostToDevice));
            if(!top.empty()) CUDA_ASSERT(cudaMemcpy(topPool[name].get(), top.data(), sizeof(T)*top.size(), cudaMemcpyHostToDevice));
            if(!buttom.empty()) CUDA_ASSERT(cudaMemcpy(buttomPool[name].get(), buttom.data(), sizeof(T)*buttom.size(), cudaMemcpyHostToDevice));
            addrLen[name] = addr.size();
            if(rank != 0) topAddrLen[name] = top.size();
            if(rank != nprocs-1) buttomAddrLen[name] = buttom.size();
        }
    }

    template <typename T>
    __host__ void transferToCPU(T* cpu, std::string name, size_t length)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        CUDA_ASSERT(cudaMemcpy(cpu, memoryPool[name].get(), sizeof(T)*length, cudaMemcpyDeviceToHost));
    }

    __host__ DataType* mallocCube(std::string name, bool create_halo=false)
    {
        assert(memoryPool.find(name) == memoryPool.end());
        DataType* p = nullptr;
        CUDA_ASSERT(cudaMalloc(&p, sizeof(DataType)*size2));
        memoryPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});

        if(create_halo)
        {
            if(rank != 0)
            {
                assert(mpiSize[rank] >= 2*M);
                CUDA_ASSERT(cudaMalloc(&p, sizeof(DataType)*3*M*YSize*PXSize));
                topPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
            }
            if(rank != nprocs-1)
            {
                assert(mpiSize[rank] >= 2*M);
                CUDA_ASSERT(cudaMalloc(&p, sizeof(DataType)*3*M*YSize*PXSize));
                buttomPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
            }
        }
        return p;
    }

    __host__ DataType* getCube(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        return (DataType*)memoryPool[name].get();
    }

    __host__ DataType* getCubeHaloTopBackup(std::string name)
    {
        if(topPool.find(name) == topPool.end()) return memoryPool[name].get();
        return (DataType*)topPool[name].get();
    }

    __host__ DataType* getCubeHaloButtomBackup(std::string name)
    {
        if(buttomPool.find(name) == buttomPool.end()) return memoryPool[name].get()+size2-3*M*YSize*PXSize;
        return (DataType*)buttomPool[name].get();
    }

    __host__ void transferCubeToGPU(std::string name, DataType* cpu)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        DataType *tmp = new DataType[ZSize*YSize*PXSize];
        memset(tmp, 0, sizeof(DataType)*ZSize*YSize*PXSize);
        for(size_t i=0;i<ZSize;i++)
            for(size_t j=0;j<YSize;j++)
                for(size_t k=0;k<XSize;k++)
                    tmp[GET(i,j,k+PADDINGL,ZSize,YSize,PADDINGL+XSize+PADDINGR)] = cpu[GET(i,j,k,ZSize,YSize,XSize)];
        CUDA_ASSERT(cudaMemcpy(memoryPool[name].get(), tmp+off2, sizeof(DataType)*size2, cudaMemcpyHostToDevice));

        delete [] tmp;
    }

    __host__ void transferCubeToCPU(DataType* cpu, std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        DataType *tmp = new DataType[ZSize*YSize*PXSize];
        CUDA_ASSERT(cudaMemcpy(tmp+off2, memoryPool[name].get(), sizeof(DataType)*size2, cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaDeviceSynchronize());
        for(size_t i=0;i<ZSize;i++)
            for(size_t j=0;j<YSize;j++)
                for(size_t k=0;k<XSize;k++)
                    cpu[GET(i,j,k,ZSize,YSize,XSize)] = tmp[GET(i,j,k+PADDINGL,ZSize,YSize,PADDINGL+XSize+PADDINGR)];
        delete [] tmp;
        size_t size = size_t(mpiSize[rank]) * YSize * XSize;
        size_t off = size_t(mpiOff[rank]) * YSize * XSize;
        if(rank == 0)
        {
            for(int i=1;i<nprocs;i++)
            {
                size_t size = size_t(mpiSize[i]) * YSize * XSize;
                size_t off = size_t(mpiOff[i]) * YSize * XSize;
                MPI::COMM_WORLD.Recv(cpu+off, size, mpi_datatype, i, 0);
            }
        }
        else
        {
            MPI::COMM_WORLD.Send(cpu+off, size, mpi_datatype, 0, 0);
        }
        MPI_COMM_WORLD_BCAST(cpu, ZSize * YSize * XSize, 0);
    }

    __host__ void commCubeHalo(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        DataType *p = memoryPool[name].get();
        size_t recv = (mpiSize[rank]+((rank==0)?0:M))*YSize*PXSize;
        size_t send = recv-M*YSize*PXSize;
        if(rank != 0) MPI::COMM_WORLD.Sendrecv(p+M*YSize*PXSize, M*YSize*PXSize, MPI::FLOAT, rank-1, rank-1, p, M*YSize*PXSize, MPI::FLOAT, rank-1, rank);
        if(rank != nprocs-1) MPI::COMM_WORLD.Sendrecv(p+send, M*YSize*PXSize, MPI::FLOAT, rank+1, rank+1, p+recv, M*YSize*PXSize, MPI::FLOAT, rank+1, rank);
    }

    __host__ void commCubeHaloBackup(std::string name)
    {
        if(rank != 0) assert(topPool.find(name) != topPool.end());
        if(rank != nprocs-1) assert(buttomPool.find(name) != buttomPool.end());

        if(rank != 0) MPI::COMM_WORLD.Sendrecv(topPool[name].get()+M*YSize*PXSize, M*YSize*PXSize, MPI::FLOAT, rank-1, rank-1, topPool[name].get(), M*YSize*PXSize, MPI::FLOAT, rank-1, rank);
        if(rank != nprocs-1) MPI::COMM_WORLD.Sendrecv(buttomPool[name].get()+M*YSize*PXSize, M*YSize*PXSize, MPI::FLOAT, rank+1, rank+1, buttomPool[name].get()+2*M*YSize*PXSize, M*YSize*PXSize, MPI::FLOAT, rank+1, rank);
    }

    __host__ void backupCubeHaloBackup(std::string name)
    {
        if(topPool.find(name) != topPool.end())
        {
            CUDA_ASSERT(cudaMemcpy(topPool[name].get(), memoryPool[name].get(), sizeof(DataType)*3*M*YSize*PXSize, cudaMemcpyDeviceToDevice));
        }

        if(buttomPool.find(name) != buttomPool.end())
        {
            CUDA_ASSERT(cudaMemcpy(buttomPool[name].get(), memoryPool[name].get()+size2-3*M*YSize*PXSize, sizeof(DataType)*3*M*YSize*PXSize, cudaMemcpyDeviceToDevice));
        }
    }

    __host__ void restoreCubeHaloBackup(std::string name)
    {
        if(topPool.find(name) != topPool.end())
        {
            CUDA_ASSERT(cudaMemcpy(memoryPool[name].get(), topPool[name].get(), sizeof(DataType)*M*YSize*PXSize, cudaMemcpyDeviceToDevice));
        }

        if(buttomPool.find(name) != buttomPool.end())
        {
            CUDA_ASSERT(cudaMemcpy(memoryPool[name].get()+size2-M*YSize*PXSize, buttomPool[name].get()+2*M*YSize*PXSize, sizeof(DataType)*M*YSize*PXSize, cudaMemcpyDeviceToDevice));
        }
    }

    template <typename Function, typename... Arguments>
    __host__ void propagate(std::string output, std::string input, bool spill, Function kernel, Arguments... args)
    {
        assert(memoryPool.find(input) != memoryPool.end());
        assert(memoryPool.find(output) != memoryPool.end());
        DataType *p0 = memoryPool[output].get(), *p1 = memoryPool[input].get();
        size_t _nx = XSize, _ny = YSize, nz = mpiSize[rank]-((rank==0)?M:0)-((rank==nprocs-1)?M:0);

        if(M <= 1)
        {
            dim3 block(64, 8);
            dim3 grid(_nx/64+1, _ny/8+1);
            if(spill)
                prop_center_nosm_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 2048/512, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            else
                prop_center_nosm_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 1, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            return ;
        }

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        if(_nx < 2*BLOCK_SIZE || _ny < 2*BLOCK_SIZE)
        {
            dim3 grid(_nx/(BLOCK_SIZE-2*M)+1, _ny/(BLOCK_SIZE-2*M)+1);
            if(spill)
                prop_small_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 2, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            else
                prop_small_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 1, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
        }
        else
        {
            size_t nx2 = (_nx - 2*M) / BLOCK_SIZE * BLOCK_SIZE + 2*M;
            size_t ny2 = (_ny - 2*M) / BLOCK_SIZE * BLOCK_SIZE + 2*M;
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            {
                dim3 grid((nx2-2*M)/BLOCK_SIZE, (ny2-2*M)/BLOCK_SIZE);
                if(spill)
                    prop_center_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 2, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                else
                    prop_center_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 1, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            }
            {
                size_t xs = ((nx2-2*M))/(BLOCK_SIZE-2*M);
                size_t ys = ((ny2-2*M))/(BLOCK_SIZE-2*M);
                size_t xt = ((_nx-2*M)+(BLOCK_SIZE-2*M)-1)/(BLOCK_SIZE-2*M);
                size_t yt = ((_ny-2*M)+(BLOCK_SIZE-2*M)-1)/(BLOCK_SIZE-2*M);
                {
                    dim3 grid(xt-xs, yt);
                    if(spill)
                        prop_halo_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 2, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1,  _nx, _ny, nz, nx2-2*M, ny2-2*M, xs, 0, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                    else
                        prop_halo_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 1, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1,  _nx, _ny, nz, nx2-2*M, ny2-2*M, xs, 0,  size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                }
                {
                    dim3 grid(xs, yt-ys);
                    if(spill)
                        prop_halo_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 2, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, nx2-2*M, ny2-2*M, 0, ys, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                    else
                        prop_halo_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 1, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, nx2-2*M, ny2-2*M, 0, ys,  size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                }
            }
        }
//        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    template <typename Function, typename... Arguments>
    __host__ void propagateHaloTopBackup(std::string output, std::string input, bool spill, Function kernel, Arguments... args)
    {
        if(topPool.find(input) != topPool.end() && topPool.find(output) != topPool.end())
        {
            DataType *p0 = topPool[output].get(), *p1 = topPool[input].get();
            size_t _nx = XSize, _ny = YSize, nz = M;
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            {
                dim3 grid(_nx/(BLOCK_SIZE-8)+1, _ny/(BLOCK_SIZE-8)+1);
                if(spill)
                    prop_small_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 2, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                else
                    prop_small_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 1, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, size_t(mpiOff[rank]-((rank == 0)?0:M))+M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            }
        }
//        CUDA_ASSERT(cudaDeviceSynchronize());

    }

    template <typename Function, typename... Arguments>
    __host__ void propagateHaloButtomBackup(std::string output, std::string input, bool spill, Function kernel, Arguments... args)
    {
        if(buttomPool.find(input) != buttomPool.end() && buttomPool.find(output) != buttomPool.end())
        {
            DataType *p0 = buttomPool[output].get(), *p1 = buttomPool[input].get();
            size_t _nx = XSize, _ny = YSize, nz = M;
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            {
                dim3 grid(_nx/(BLOCK_SIZE-8)+1, _ny/(BLOCK_SIZE-8)+1);
                if(spill)
                    prop_small_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 2, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, mpiOff[rank+1]-M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                else
                    prop_small_kernel<DataType, gpu_size_t, gpu_signed_size_t, M, 1, PADDINGL, PADDINGR, BLOCK_SIZE, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, mpiOff[rank+1]-M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            }
        }
//        CUDA_ASSERT(cudaDeviceSynchronize());

    }

    template <typename Function, typename... Arguments>
    __host__ void inject(size_t length, Function kernel, Arguments... args)
    {
        inject_kernel<gpu_size_t, gpu_signed_size_t, Function, Arguments...><<<length/THREADS+1, THREADS>>>(length, kernel, args...);
//        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    template <typename Function, typename... Arguments>
    __host__ void filt(Function kernel, Arguments... args)
    {
        dim3 grid(XSize/THREADS+1, YSize, mpiSize[rank]);
        filt_kernel<gpu_size_t, gpu_signed_size_t, Function, Arguments...><<<grid, THREADS>>>(XSize, YSize, mpiOff[rank], (rank == 0)?0:M, PADDINGL, PADDINGR, kernel, args...);
//        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    template <typename Function, typename... Arguments>
    __host__ void filtHaloTopBackup(Function kernel, Arguments... args)
    {
        if(rank == 0) return;
        dim3 grid(XSize/THREADS+1, YSize, 2*M);
        filt_kernel<gpu_size_t, gpu_signed_size_t, Function, Arguments...><<<grid, THREADS>>>(XSize, YSize, mpiOff[rank], M, PADDINGL, PADDINGR, kernel, args...);
//        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    template <typename Function, typename... Arguments>
    __host__ void filtHaloButtomBackup(Function kernel, Arguments... args)
    {
        if(rank == nprocs-1) return;
        dim3 grid(XSize/THREADS+1, YSize, 2*M);
        filt_kernel<gpu_size_t, gpu_signed_size_t, Function, Arguments...><<<grid, THREADS>>>(XSize, YSize, mpiOff[rank+1]-2*M, 0, PADDINGL, PADDINGR, kernel, args...);
//        CUDA_ASSERT(cudaDeviceSynchronize());
    }
};


