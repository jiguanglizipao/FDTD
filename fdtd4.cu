#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <random>
#include "stencil.h"
const size_t nz = 512;
const size_t ny = 512;
const size_t nx = 512;
const size_t na = 10000;
#define GET(z, y, x, nz, ny, nx) ((z)*((ny)*(nx))+(y)*(nx)+(x))
int main(const int argc, const char* argv[])
{
    Stencil<float, nz, ny, nx, 4> stencil;
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
    stencil.mallocCube("p0", true);
    stencil.mallocCube("p1", true);
    stencil.mallocCube("sum");
    stencil.mallocCube("vel");
    stencil.malloc<size_t>("addr", na, true);
    stencil.transferCubeToGPU("p0", cpu_p0);
    stencil.transferCubeToGPU("p1", cpu_p1);
    stencil.transferCubeToGPU("sum", cpu_sum);
    stencil.transferCubeToGPU("vel", vel);
    stencil.transferToGPU("addr", addr, na);
    auto prop_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *output, float* zl, int sz, float *yl, int sy, float *xl, int sx, float*vel, float scal)
    {
        output[addr] = vel[addr]*scal*(zl[0]+
                       1.0f*(xl[1*sx]+xl[-1*sx])+
                       1.0f*(xl[2*sx]+xl[-2*sx])+
                       1.0f*(xl[3*sx]+xl[-3*sx])+
                       1.0f*(xl[4*sx]+xl[-4*sx])+
                       2.0f*(yl[1*sy]+yl[-1*sy])+
                       2.0f*(yl[2*sy]+yl[-2*sy])+
                       2.0f*(yl[3*sy]+yl[-3*sy])+
                       2.0f*(yl[4*sy]+yl[-4*sy])+
                       3.0f*(zl[1*sz]+zl[-1*sz])+
                       3.0f*(zl[2*sz]+zl[-2*sz])+
                       3.0f*(zl[3*sz]+zl[-3*sz])+
                       3.0f*(zl[4*sz]+zl[-4*sz])
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
    stencil.barrier();
    gettimeofday(&start, NULL);
    std::string s0 = "p0", s1 = "p1";
    for(size_t t=0;t<1000;t++)
    {
        stencil.backupCubeHaloBackup(s0);
        stencil.backupCubeHaloBackup(s1);
        stencil.propagateHaloTopBackup(s0, s1, true, prop_kernel, stencil.getCubeHaloTopBackup("vel"), 1.0/(6*8+1));
        stencil.propagateHaloButtomBackup(s0, s1, true, prop_kernel, stencil.getCubeHaloButtomBackup("vel"), 1.0/(6*8+1));
        stencil.sync();
        stencil.propagate(s0, s1, true, prop_kernel, stencil.getCube("vel"), 1.0/(6*8+1));
        stencil.commCubeHaloBackup(s0);
        stencil.commCubeHaloBackup(s1);
        stencil.sync();
        stencil.restoreCubeHaloBackup(s0);
        stencil.restoreCubeHaloBackup(s1);
        stencil.sync();
        std::swap(s0, s1);
    }
    gettimeofday(&end, NULL);
    printf("GPU time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    memset(cpu_p0, 0, sizeof(float)*nz*ny*nx);
    memset(cpu_p1, 0, sizeof(float)*nz*ny*nx);
    memset(cpu_sum, 0, sizeof(float)*nz*ny*nx);
    stencil.transferCubeToCPU(cpu_p0, s0);
    stencil.transferCubeToCPU(cpu_p1, s1);
    stencil.transferCubeToCPU(cpu_sum, "sum");
    return 0;
}
