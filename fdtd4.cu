#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <random>
#include "stencil.h"
const size_t nz = 512;
const size_t ny = 512;
const size_t nx = 512;
const size_t na = 10000;
#define GET(z, y, x, nz, ny, nx) ((z)*((ny)*(nx))+(y)*(nx)+(x))
typedef uint32_t gpu_size_t;
typedef int32_t gpu_signed_size_t;
int main(const int argc, const char* argv[])
{
    Stencil<float, nz, ny, nx, 4, true, gpu_size_t, gpu_signed_size_t> stencil;
    float *p0 = new float[nz*ny*nx];
    float *p1 = new float[nz*ny*nx];
    float *vel = new float[nz*ny*nx];
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
            }
    }
    for(size_t i=0;i<na;i++)
    {
        size_t x = rand()%nx, y = rand()%ny, z = rand()%nz;
        addr[i] = GET(z,y,x,nz,ny,nx);
    }
    float *gpu_p0 = new float[nz*ny*nx];
    float *gpu_p1 = new float[nz*ny*nx];
    memcpy(gpu_p0, p0, sizeof(float)*nz*ny*nx);
    memcpy(gpu_p1, p1, sizeof(float)*nz*ny*nx);
    
    std::chrono::time_point<std::chrono::system_clock> start, end;
#ifndef SKIP_CPU
    start = std::chrono::system_clock::now();
    for(size_t t=0;t<100;t++)
    {
        #pragma omp parallel for
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
                                              3.0*(p1[GET(i+4,j,k,nz,ny,nx)]+p1[GET(i-4,j,k,nz,ny,nx)])
                                              )
                                              -p0[GET(i,j,k,nz,ny,nx)];
        if(stencil.getRank() == 0) printf("loop %lu\n", t);
        std::swap(p0, p1);
    }
    end = std::chrono::system_clock::now();
    printf("CPU time %.6lfs\n", 1e-6*(std::chrono::time_point_cast<std::chrono::microseconds>(end)-std::chrono::time_point_cast<std::chrono::microseconds>(start)));
#endif

    stencil.mallocCube("p0", true);
    stencil.mallocCube("p1", true);
    stencil.mallocCube("vel");
    stencil.malloc<size_t>("addr", na, true);
    stencil.transferCubeToGPU("p0", gpu_p0);
    stencil.transferCubeToGPU("p1", gpu_p1);
    stencil.transferCubeToGPU("vel", vel);
    stencil.transferToGPU("addr", addr, na);
    auto prop_kernel = [=] __device__ (gpu_size_t z, gpu_size_t y, gpu_size_t x, gpu_size_t addr, float *output, float* zl, gpu_signed_size_t sz, float *yl, gpu_signed_size_t sy, float *xl, gpu_signed_size_t sx, float*vel, float scal)
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
    auto inject_kernel = [=] __device__ (gpu_size_t i, float *output, gpu_size_t *addr, float add)
    {
        atomicAdd(&output[addr[i]], add);
    };
    auto filt_kernel = [=] __device__ (gpu_size_t z, gpu_size_t y, gpu_size_t x, gpu_size_t addr, float *output)
    {
        if(z == nz/2 || y == ny/3 || x == nx/4)
            output[addr] *= 0.9f;
    };
    stencil.barrier();
    start = std::chrono::system_clock::now();
    std::string s0 = "p0", s1 = "p1";
    for(size_t t=0;t<100;t++)
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
    end = std::chrono::system_clock::now();
    printf("GPU time %.6lfs\n", 1e-6*(std::chrono::time_point_cast<std::chrono::microseconds>(end)-std::chrono::time_point_cast<std::chrono::microseconds>(start)));
    memset(gpu_p0, 0, sizeof(float)*nz*ny*nx);
    memset(gpu_p1, 0, sizeof(float)*nz*ny*nx);
    stencil.transferCubeToCPU(gpu_p0, s0);
    stencil.transferCubeToCPU(gpu_p1, s1);
    size_t rank = stencil.getRank();
#ifndef SKIP_CPU
    #pragma omp parallel for
    for(size_t i=0;i<nz;i++)
        for(size_t j=0;j<ny;j++)
            for(size_t k=0;k<nx;k++)
            {
                float dp0 = fabs(p0[GET(i,j,k,nz,ny,nx)]-gpu_p0[GET(i,j,k,nz,ny,nx)]),
                      dp1 = fabs(p1[GET(i,j,k,nz,ny,nx)]-gpu_p1[GET(i,j,k,nz,ny,nx)]);

                if((dp0/fabs(gpu_p0[GET(i,j,k,nz,ny,nx)])>1e-2 && dp0>1e-3) ||
                   (dp1/fabs(gpu_p0[GET(i,j,k,nz,ny,nx)])>1e-2 && dp0>1e-3))
                    fprintf(stderr, "rank = %lu, [%lu][%lu][%lu]: %lf %lf, %lf %lf\n", rank, i, j, k, p0[GET(i,j,k,nz,ny,nx)], gpu_p0[GET(i,j,k,nz,ny,nx)], p1[GET(i,j,k,nz,ny,nx)], gpu_p1[GET(i,j,k,nz,ny,nx)]);
            }
#endif 
    return 0;
}
