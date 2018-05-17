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
    Stencil<float, nz, ny, nx, 1> stencil;
    float *p0 = new float[nz*ny*nx];
    float *p1 = new float[nz*ny*nx];

    #pragma omp parallel for
    for(size_t i=0;i<nz;i++)
    {
        auto rand = std::mt19937_64(19950918+i);
        for(size_t j=0;j<ny;j++)
            for(size_t k=0;k<nx;k++)
            {
                p0[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
                p1[GET(i,j,k,nz,ny,nx)] = float(rand())/rand.max();
            }
    }
    struct timeval start, end;
    stencil.mallocCube("p0", true);
    stencil.mallocCube("p1", true);
    stencil.transferCubeToGPU("p0", p0);
    stencil.transferCubeToGPU("p1", p1);
    auto prop_kernel = [=] __device__ (size_t z, size_t y, size_t x, size_t addr, float *output, float* zl, int sz, float *yl, int sy, float *xl, int sx)
    {
        output[addr] = (0.01f*zl[0]+0.02f*xl[1*sx]+0.03f*xl[-1*sx]+0.04f*yl[1*sy]+0.05f*yl[-1*sy]+0.06f*zl[1*sz]+0.07f*zl[-1*sz]);
    };
    stencil.barrier();
    gettimeofday(&start, NULL);
    std::string s0 = "p0", s1 = "p1";
    for(size_t t=0;t<1000;t++)
    {
        stencil.backupCubeHaloBackup(s0);
        stencil.backupCubeHaloBackup(s1);
        stencil.propagateHaloTopBackup(s0, s1, true, prop_kernel);
        stencil.propagateHaloButtomBackup(s0, s1, true, prop_kernel);
        stencil.sync();
        stencil.propagate(s0, s1, true, prop_kernel);
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
    stencil.transferCubeToCPU(p0, s0);
    stencil.transferCubeToCPU(p1, s1);
    return 0;
}
