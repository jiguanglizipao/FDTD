#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <random>
#include "fdtd.h"
const size_t nz = 512;
const size_t ny = 512;
const size_t nx = 512;
const size_t na = 10000;
#define GET(z, y, x, nz, ny, nx) ((z)*((ny)*(nx))+(y)*(nx)+(x))
int main(const int argc, const char* argv[])
{
    FDTD<float, nz, ny, nx, 4> fdtd;
    float *p0 = new float[nz*ny*nx];
    float *p1 = new float[nz*ny*nx];
    float *vel = new float[nz*ny*nx];
    float *sum = new float[nz*ny*nx];
    size_t *addr = new size_t[na];

    #pragma omp parallel for
    for(size_t i=0;i<nz;i++)
    {
        auto rand = std::mt19937_64(19950918+i);
        for(size_t j=0;j<ny;j++)
            for(size_t k=0;k<nx;k++)
            {
                p0[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
                p1[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
                vel[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
                sum[GET(i,j,k,nz,ny,nx)] = 0.0;
            }
    }
    for(size_t i=0;i<na;i++)
    {
        size_t x = rand()%nx, y = rand()%ny, z = rand()%nz;
        addr[i] = GET(z,y,x,nz,ny,nx);
    }
    float *cpu_p0 = new float[nz*ny*nx];
    float *cpu_p1 = new float[nz*ny*nx];
    float *cpu_sum = new float[nz*ny*nx];
    memcpy(cpu_p0, p0, sizeof(float)*nz*ny*nx);
    memcpy(cpu_p1, p1, sizeof(float)*nz*ny*nx);
    memcpy(cpu_sum, sum, sizeof(float)*nz*ny*nx);
    
    struct timeval start, end;
    fdtd.mallocCube("p0", true);
    fdtd.mallocCube("p1", true);
    fdtd.mallocCube("sum");
    fdtd.mallocCube("vel");
    fdtd.malloc<size_t>("addr", na, true);
    fdtd.transferCubeToGPU("p0", cpu_p0);
    fdtd.transferCubeToGPU("p1", cpu_p1);
    fdtd.transferCubeToGPU("sum", cpu_sum);
    fdtd.transferCubeToGPU("vel", vel);
    fdtd.transferToGPU("addr", addr, na);
    auto prop_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *output, float* zl, float *yl, float *xl, float*vel, float scal)
    {
        output[addr] = vel[addr]*scal*(zl[0]+
                       1.0f*(xl[1]+xl[-1])+
                       1.0f*(xl[2]+xl[-2])+
                       1.0f*(xl[3]+xl[-3])+
                       1.0f*(xl[4]+xl[-4])+
                       2.0f*(yl[1]+yl[-1])+
                       2.0f*(yl[2]+yl[-2])+
                       2.0f*(yl[3]+yl[-3])+
                       2.0f*(yl[4]+yl[-4])+
                       3.0f*(zl[1]+zl[-1])+
                       3.0f*(zl[2]+zl[-2])+
                       3.0f*(zl[3]+zl[-3])+
                       3.0f*(zl[4]+zl[-4])
                       )
                       -output[addr];
    };
    auto inject_kernel = [=] __device__ (size_t i, float *output, size_t *addr, float add)
    {
        atomicAdd(&output[addr[i]], add);
    };
    auto filt_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *output)
    {
        if(z == nz/2 || y == ny/3 || x == nx/4)
            output[addr] *= 0.9f;
    };
    auto mul_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *sum, float *p0, float *p1)
    {
        if(z >= 4 && y >= 4 && x >= 4 && z < nz-4 && y < ny-4 && x < nx-4)
            sum[addr] += p0[addr]*p1[addr];
    };
    fdtd.barrier();
    gettimeofday(&start, NULL);
    std::string s0 = "p0", s1 = "p1";
    for(size_t t=0;t<1000;t++)
    {
        fdtd.backupCubeHaloBackup(s0);
        fdtd.backupCubeHaloBackup(s1);
        fdtd.propagateHaloTopBackup(s0, s1, true, prop_kernel, fdtd.getCubeHaloTopBackup("vel"), 1.0/(6*8+1));
        fdtd.propagateHaloButtomBackup(s0, s1, true, prop_kernel, fdtd.getCubeHaloButtomBackup("vel"), 1.0/(6*8+1));
        fdtd.sync();
        fdtd.propagate(s0, s1, true, prop_kernel, fdtd.getCube("vel"), 1.0/(6*8+1));
        fdtd.commCubeHaloBackup(s0);
        fdtd.commCubeHaloBackup(s1);
        fdtd.sync();
        fdtd.restoreCubeHaloBackup(s0);
        fdtd.restoreCubeHaloBackup(s1);
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
}
