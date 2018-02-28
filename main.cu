#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <random>
#include "fdtd.h"
const size_t nz = 59;//8;
const size_t ny = 53;//2;
const size_t nx = 51;//2;
const size_t na = 10000;
#define GET(z, y, x, nz, ny, nx) ((z)*((ny)*(nx))+(y)*(nx)+(x))
int main(const int argc, const char* argv[])
{
    auto rand = std::mt19937_64(19950918);
    FDTD<float, nz, ny, nx, 4> fdtd;
    float *p0 = new float[nz*ny*nx];
    float *p1 = new float[nz*ny*nx];
    float *vel = new float[nz*ny*nx];
    float *sum = new float[nz*ny*nx];
    size_t *addr = new size_t[na];
    size_t *gpu_addr = new size_t[na];
    for(size_t i=0;i<nz;i++)
        for(size_t j=0;j<ny;j++)
            for(size_t k=0;k<nx;k++)
            {
                p0[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
                p1[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
                vel[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
                sum[GET(i,j,k,nz,ny,nx)] = 0.0;
            }
    for(size_t i=0;i<na;i++)
    {
        size_t x = rand()%nx, y = rand()%ny, z = rand()%nz;
        addr[i] = GET(z,y,x,nz,ny,nx);
        gpu_addr[i] = fdtd.addrTrans(z,y,x);
    }
    float *cpu_p0 = new float[nz*ny*nx];
    float *cpu_p1 = new float[nz*ny*nx];
    float *cpu_sum = new float[nz*ny*nx];
    memcpy(cpu_p0, p0, sizeof(float)*nz*ny*nx);
    memcpy(cpu_p1, p1, sizeof(float)*nz*ny*nx);
    memcpy(cpu_sum, sum, sizeof(float)*nz*ny*nx);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for(size_t t=0;t<10;t++)
    {
        for(size_t i=4;i<nz-4;i++)
            for(size_t j=4;j<ny-4;j++)
                for(size_t k=4;k<nx-4;k++)
                    p0[GET(i,j,k,nz,ny,nx)] = vel[GET(i,j,k,nz,ny,nx)]*1./(6*8+1)*(p1[GET(i,j,k,nz,ny,nx)]+
                                                     1.0*(p1[GET(i,j,k+1,nz,ny,nx)]+p1[GET(i,j,k-1,nz,ny,nx)])+
                                                     1.0*(p1[GET(i,j,k+2,nz,ny,nx)]+p1[GET(i,j,k-2,nz,ny,nx)])+
                                                     1.0*(p1[GET(i,j,k+3,nz,ny,nx)]+p1[GET(i,j,k-3,nz,ny,nx)])+
                                                     1.0*(p1[GET(i,j,k+4,nz,ny,nx)]+p1[GET(i,j,k-4,nz,ny,nx)])+
                                                     2.0*(p1[GET(i,j+1,k,nz,ny,nx)]+p1[GET(i,j-1,k,nz,ny,nx)])+
                                                     2.0*(p1[GET(i,j+2,k,nz,ny,nx)]+p1[GET(i,j-2,k,nz,ny,nx)])+
                                                     2.0*(p1[GET(i,j+3,k,nz,ny,nx)]+p1[GET(i,j-3,k,nz,ny,nx)])+
                                                     2.0*(p1[GET(i,j+4,k,nz,ny,nx)]+p1[GET(i,j-4,k,nz,ny,nx)])+
                                                     3.0*(p1[GET(i+1,j,k,nz,ny,nx)]+p1[GET(i-1,j,k,nz,ny,nx)])+
                                                     3.0*(p1[GET(i+2,j,k,nz,ny,nx)]+p1[GET(i-2,j,k,nz,ny,nx)])+
                                                     3.0*(p1[GET(i+3,j,k,nz,ny,nx)]+p1[GET(i-3,j,k,nz,ny,nx)])+
                                                     3.0*(p1[GET(i+4,j,k,nz,ny,nx)]+p1[GET(i-4,j,k,nz,ny,nx)]))
                                                     -p0[GET(i,j,k,nz,ny,nx)];
        for(size_t i=0;i<na;i++)p0[addr[i]] += 1;
        for(size_t i=0;i<nz;i++)
            for(size_t j=0;j<ny;j++)
                for(size_t k=0;k<nx;k++)
                    if(i == nz/2 || j == ny/3 || k == nx/4)
                        p0[GET(i,j,k,nz,ny,nx)] *= 0.9;
        for(size_t i=4;i<nz-4;i++)
            for(size_t j=4;j<ny-4;j++)
                for(size_t k=4;k<nx-4;k++)
                    sum[GET(i,j,k,nz,ny,nx)] += p0[GET(i,j,k,nz,ny,nx)]*p1[GET(i,j,k,nz,ny,nx)];
        printf("loop %lu\n", t);
        std::swap(p0, p1);
    }
    gettimeofday(&end, NULL);
    printf("CPU time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    fdtd.mallocCube("p0");
    fdtd.mallocCube("p1");
    fdtd.mallocCube("sum");
    fdtd.mallocCube("vel");
    fdtd.malloc<size_t>("addr", na);
    fdtd.transferCubeToGPU("p0", cpu_p0);
    fdtd.transferCubeToGPU("p1", cpu_p1);
    fdtd.transferCubeToGPU("sum", cpu_sum);
    fdtd.transferCubeToGPU("vel", vel);
    fdtd.transferToGPU("addr", gpu_addr, na);
    auto prop_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *output, float* zl, float *yl, float *xl, float*vel, float scal)
    {
        output[addr] = vel[addr]*scal*(zl[0]+
                       1.0*(xl[1]+xl[-1])+
                       1.0*(xl[2]+xl[-2])+
                       1.0*(xl[3]+xl[-3])+
                       1.0*(xl[4]+xl[-4])+
                       2.0*(yl[1]+yl[-1])+
                       2.0*(yl[2]+yl[-2])+
                       2.0*(yl[3]+yl[-3])+
                       2.0*(yl[4]+yl[-4])+
                       3.0*(zl[1]+zl[-1])+
                       3.0*(zl[2]+zl[-2])+
                       3.0*(zl[3]+zl[-3])+
                       3.0*(zl[4]+zl[-4]))
                       -output[addr];
    };
    auto inject_kernel = [=] __device__ (size_t i, float *output, size_t *addr, float add)
    {
        if(addr[i] != -1) atomicAdd(&output[addr[i]], add);
    };
    auto filt_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *output)
    {
        if(z == nz/2 || y == ny/3 || x == nx/4)
            output[addr] *= 0.9;
    };
    auto mul_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *sum, float *p0, float *p1)
    {
        if(z >= 4 && y >= 4 && x >= 4 && z < nz-4 && y < ny-4 && x < nx-4)
            sum[addr] += p0[addr]*p1[addr];
    };
    gettimeofday(&start, NULL);
    std::string s0 = "p0", s1 = "p1";
    for(size_t t=0;t<10;t++)
    {
        fdtd.propagate(s0, s1, true, prop_kernel, fdtd.getCube("vel"), 1.0/(6*8+1));
        fdtd.inject(na, inject_kernel, fdtd.getCube(s0), fdtd.get<size_t>("addr"), 1.0);
        fdtd.filt(filt_kernel, fdtd.getCube(s0));
        fdtd.filt(mul_kernel, fdtd.getCube("sum"), fdtd.getCube(s0), fdtd.getCube(s1));
        fdtd.commCubeHalo(s0);
        fdtd.commCubeHalo(s1);
        fdtd.sync();
        std::swap(s0, s1);
    }
    gettimeofday(&end, NULL);
    printf("GPU time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    memset(cpu_p0, 0, sizeof(float)*nz*ny*nx);
    memset(cpu_p1, 0, sizeof(float)*nz*ny*nx);
    memset(cpu_sum, 0, sizeof(float)*nz*ny*nx);
    fdtd.transferCubeToCPU(cpu_p0, s0);
    fdtd.transferCubeToCPU(cpu_p1, s1);
    fdtd.transferCubeToCPU(cpu_sum, "sum");

    auto rank = MPI::COMM_WORLD.Get_rank();
    for(size_t i=0;i<nz;i++)
        for(size_t j=0;j<ny;j++)
            for(size_t k=0;k<nx;k++)
            {
                if(fabs(p0[GET(i,j,k,nz,ny,nx)]-cpu_p0[GET(i,j,k,nz,ny,nx)])>1e-5 ||
                   fabs(p1[GET(i,j,k,nz,ny,nx)]-cpu_p1[GET(i,j,k,nz,ny,nx)])>1e-5 ||
                   fabs(sum[GET(i,j,k,nz,ny,nx)]-cpu_sum[GET(i,j,k,nz,ny,nx)])>1e-5)
                    fprintf(stderr, "rank = %d, [%lu][%lu][%lu]: %lf %lf, %lf %lf, %lf %lf\n", rank, i, j, k, p0[GET(i,j,k,nz,ny,nx)], cpu_p0[GET(i,j,k,nz,ny,nx)], p1[GET(i,j,k,nz,ny,nx)], cpu_p1[GET(i,j,k,nz,ny,nx)], sum[GET(i,j,k,nz,ny,nx)], cpu_sum[GET(i,j,k,nz,ny,nx)]);
            }
    cudaDeviceSynchronize();
}
