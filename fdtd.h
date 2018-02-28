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

template <typename DataType, size_t M, size_t blocks, size_t PADDINGL, size_t PADDINGR, typename Function, typename... Arguments>
__launch_bounds__(1024, blocks)
__global__ static void prop_center_kernel(DataType *p0, DataType *p1, size_t _nx, size_t _ny, size_t _nz, size_t offz, size_t offaddr, Function kernel, Arguments... args)
{
    const size_t BLOCK_SIZE = 32;
    __shared__ DataType p1s[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M];
    __shared__ DataType p1st[BLOCK_SIZE+2*M][BLOCK_SIZE+2*M];

    size_t ig = (blockIdx.x * blockDim.x + threadIdx.x + M + PADDINGL);
    size_t jg = (blockIdx.y * blockDim.y + threadIdx.y + M);

    size_t il = threadIdx.x + M;
    size_t jl = threadIdx.y + M; 

    DataType p1z[2*M+1];
    size_t _n12 = (_nx+PADDINGL+PADDINGR)*_ny;
    size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg+offaddr;
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
        kernel(yl+offz, jg, ig-PADDINGL, addr, p0, p1z+M, &p1st[il][jl], &p1s[jl][il], args...);
        __syncthreads();
        addr+=_n12;
        if(threadIdx.y < 4*M)
            p1st[ir][jr] = p1s[jr][ir] = p1[addr + offr];
    }
}

template <typename DataType, size_t M, size_t blocks, size_t PADDINGL, size_t PADDINGR,  typename Function, typename... Arguments>
__launch_bounds__(1024, blocks)
__global__ static void prop_halo_kernel(DataType *p0, DataType *p1, size_t _nx, size_t _ny, size_t _nz, size_t nx2, size_t ny2, size_t offx, size_t offy, size_t offz, size_t offaddr, Function kernel, Arguments... args)
{
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
    size_t addr = ig+(_nx+PADDINGL+PADDINGR)*jg+offaddr;
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
            kernel(yl+offz, jg, ig-PADDINGL, addr, p0, p1z+M, &p1st[il][jl], &p1s[jl][il], args...);
        __syncthreads();
        addr+=_n12;
    }
}

template <typename Function, typename... Arguments>
 __global__ void inject_kernel(size_t length, Function kernel, Arguments... args) 
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= length) return;
    kernel(i, args...);
}

template <typename Function, typename... Arguments>
 __global__ void filt_kernel(size_t _nx, size_t _ny, size_t zbase, size_t zoff, size_t PADDINGL, size_t PADDINGR, Function kernel, Arguments... args) 
{
    size_t x = blockDim.x*blockIdx.x+threadIdx.x;
    if(x >= _nx)return;
    size_t y = blockIdx.y;
    size_t z = blockIdx.z;
    size_t addr = (z+zoff)*(_ny*(_nx+PADDINGL+PADDINGR))+y*(_nx+PADDINGL+PADDINGR)+x+PADDINGL;

    kernel(zbase+z, y, x, addr, args...);
}

template <typename DataType, size_t ZSize, size_t YSize, size_t XSize, size_t M, size_t PADDINGL=M, size_t PADDINGR=(XSize%32>(32-M))?(64-M-XSize%32):(32-M-XSize%32), size_t BLOCK_SIZE=32, size_t THREADS=256>
class FDTD
{
private:
    std::map<std::string, std::shared_ptr<DataType>> memoryPool;
    size_t nprocs, rank, size2, off2, off, size, PXSize;
    std::vector<size_t> mpiSize, mpiOff;
    MPI::Datatype mpi_datatype;

public:
   __host__ FDTD()
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

    __host__ ~FDTD()
    {
        MPI::Finalize();
    }

    __host__ void MPI_COMM_WORLD_BCAST(DataType* data, size_t size, int root)
    {
        const size_t MAXINT = 1024*1024*512;
        size_t off=0;
        for(;size>MAXINT;size-=MAXINT,off+=MAXINT)
            MPI::COMM_WORLD.Bcast(data+off, MAXINT, mpi_datatype, root);
        MPI::COMM_WORLD.Bcast(data+off, size, mpi_datatype, root);
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

    __host__ void sync()
    {
        assert(cudaDeviceSynchronize() == cudaSuccess);
    }

    __host__ void free(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        memoryPool.erase(memoryPool.find(name));
    }

    template <typename T=DataType>
    __host__ T* get(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        return (T*)memoryPool[name].get();
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
        CUDA_ASSERT(cudaMalloc(&p, sizeof(DataType)*size2));
        memoryPool[name] = std::shared_ptr<DataType> ((DataType*)p, [](DataType* p){CUDA_ASSERT(cudaFree(p));});
        return p;
    }

    __host__ DataType* getCube(std::string name)
    {
        assert(memoryPool.find(name) != memoryPool.end());
        return (DataType*)memoryPool[name].get();
    }

    size_t GET(size_t z, size_t y, size_t x, size_t nz, size_t ny, size_t nx)
    {
        return z*(ny*nx)+y*nx+x;
    }

    size_t addrTrans(size_t z, size_t y, size_t x)
    {
        if(z < mpiOff[rank] || z >= mpiOff[rank+1])
            return -1;
        else 
            return (z-(mpiOff[rank]-((rank == 0)?0:M)))*(YSize*PXSize)+y*PXSize+(x+PADDINGL);
    }

    __host__ void transferCubeToGPU(std::string name, DataType* cpu)
    {
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
        DataType *tmp = new DataType[ZSize*YSize*PXSize];
        CUDA_ASSERT(cudaMemcpy(tmp+off2, memoryPool[name].get(), sizeof(DataType)*size2, cudaMemcpyDeviceToHost));
        for(size_t i=0;i<ZSize;i++)
            for(size_t j=0;j<YSize;j++)
                for(size_t k=0;k<XSize;k++)
                    cpu[GET(i,j,k,ZSize,YSize,XSize)] = tmp[GET(i,j,k+PADDINGL,ZSize,YSize,PADDINGL+XSize+PADDINGR)];
        delete [] tmp;
        assert(cudaDeviceSynchronize() == cudaSuccess);
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
        DataType *p = memoryPool[name].get();
        size_t recv = (mpiSize[rank]+((rank==0)?0:M))*YSize*PXSize;
        size_t send = recv-M*YSize*PXSize;
        if(rank != 0) MPI::COMM_WORLD.Sendrecv(p+M*YSize*PXSize, M*YSize*PXSize, MPI::FLOAT, rank-1, rank-1, p, M*YSize*PXSize, MPI::FLOAT, rank-1, rank);
        if(rank != nprocs-1) MPI::COMM_WORLD.Sendrecv(p+send, M*YSize*PXSize, MPI::FLOAT, rank+1, rank+1, p+recv, M*YSize*PXSize, MPI::FLOAT, rank+1, rank);
    }

    template <typename Function, typename... Arguments>
    __host__ void propagate(std::string output, std::string input, bool spill, Function kernel, Arguments... args)
    {
        assert(memoryPool.find(input) != memoryPool.end());
        assert(memoryPool.find(output) != memoryPool.end());
        DataType *p0 = memoryPool[output].get(), *p1 = memoryPool[input].get();
        size_t _nx = XSize, _ny = YSize, nz = mpiSize[rank]-((rank==0)?M:0)-((rank==nprocs-1)?M:0);
        size_t nx2 = (_nx - 2*M) / BLOCK_SIZE * BLOCK_SIZE + 2*M;
        size_t ny2 = (_ny - 2*M) / BLOCK_SIZE * BLOCK_SIZE + 2*M;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        {
            dim3 grid((nx2-2*M)/BLOCK_SIZE, (ny2-2*M)/BLOCK_SIZE);
            if(spill)
                prop_center_kernel<DataType, M, 2, PADDINGL, PADDINGR, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            else
                prop_center_kernel<DataType, M, 1, PADDINGL, PADDINGR, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
        }
        {
            size_t xs = ((nx2-2*M))/(BLOCK_SIZE-2*M);
            size_t ys = ((ny2-2*M))/(BLOCK_SIZE-2*M);
            size_t xt = ((_nx-2*M)+(BLOCK_SIZE-2*M)-1)/(BLOCK_SIZE-2*M);
            size_t yt = ((_ny-2*M)+(BLOCK_SIZE-2*M)-1)/(BLOCK_SIZE-2*M);
            {
                dim3 grid(xt-xs, yt);
                if(spill)
                    prop_halo_kernel<DataType, M, 2, PADDINGL, PADDINGR, Function, Arguments...><<<grid, block>>>(p0, p1,  _nx, _ny, nz, nx2-2*M, ny2-2*M, xs, 0, M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                else
                    prop_halo_kernel<DataType, M, 1, PADDINGL, PADDINGR, Function, Arguments...><<<grid, block>>>(p0, p1,  _nx, _ny, nz, nx2-2*M, ny2-2*M, xs, 0, M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            }
            {
                dim3 grid(xs, yt-ys);
                if(spill)
                    prop_halo_kernel<DataType, M, 2, PADDINGL, PADDINGR, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, nx2-2*M, ny2-2*M, 0, ys, M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
                else
                    prop_halo_kernel<DataType, M, 1, PADDINGL, PADDINGR, Function, Arguments...><<<grid, block>>>(p0, p1, _nx, _ny, nz, nx2-2*M, ny2-2*M, 0, ys, M, M*_ny*(_nx+PADDINGL+PADDINGR), kernel, args...);
            }
        }
        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    template <typename Function, typename... Arguments>
    __host__ void inject(size_t length, Function kernel, Arguments... args)
    {
        inject_kernel<Function, Arguments...><<<length/THREADS+1, THREADS>>>(length, kernel, args...);
        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    template <typename Function, typename... Arguments>
    __host__ void filt(Function kernel, Arguments... args)
    {
        dim3 grid(XSize/THREADS+1, YSize, mpiSize[rank]);
        filt_kernel<Function, Arguments...><<<grid, THREADS>>>(XSize, YSize, mpiOff[rank], (rank == 0)?0:M, PADDINGL, PADDINGR, kernel, args...);
        CUDA_ASSERT(cudaDeviceSynchronize());
    }
};


