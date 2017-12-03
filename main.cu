#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include "fdtd.h"
const int nz = 512;
const int ny = 512;
const int nx = 512;
#define GET(z, y, x, nz, ny, nx) ((z)*((ny)*(nx))+(y)*(nx)+(x))
int main(const int argc, const char* argv[])
{
//    int x = atoi(argv[1]);
//    int y = atoi(argv[2]);
//    auto lambda1 = [=] __device__ (int x, int y)
//    {
//        for(int i=0;i<y;i++)x=x*x;
//        return x;
//    };
//    auto lambda2 = [=] __device__ (int x, int y)
//    {
//        return x+y;
//    };
//    kernel<decltype(&function1), function1><<<1,1>>>(x, y);
//    kernel<decltype(&function2), function2><<<1,1>>>(x, y);
//    kernel<<<1,1>>>(class1(), x, y);
//    kernel<<<1,1>>>(class2(), x, y);
//    kernel<<<1,1>>>(lambda1, x, y);
//    kernel<<<1,1>>>(lambda2, x, y);
//    cudaDeviceSynchronize();
    float *p0 = new float[nz*ny*nx];
    float *p1 = new float[nz*ny*nx];
    for(size_t i=0;i<nz;i++)
        for(size_t j=0;j<ny;j++)
            for(size_t k=0;k<nx;k++)
            {
                p0[GET(i,j,k,nz,ny,nx)] = float(rand())/RAND_MAX;
                p1[GET(i,j,k,nz,ny,nx)] = float(rand())/RAND_MAX;
            }
    float *cpu_p0 = new float[nz*ny*nx];
    float *cpu_p1 = new float[nz*ny*nx];
    memcpy(cpu_p0, p0, sizeof(float)*nz*ny*nx);
    memcpy(cpu_p1, p1, sizeof(float)*nz*ny*nx);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for(size_t t=0;t<20;t++)
    {
        for(size_t i=4;i<nz-4;i++)
            for(size_t j=4;j<ny-4;j++)
                for(size_t k=4;k<nx-4;k++)
                {
                    p0[GET(i,j,k,nz,ny,nx)] = 1./(6*8+1)*(p1[GET(i,j,k,nz,ny,nx)]+
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
                                                     );
                }
        printf("loop %d\n", t);
        std::swap(p0, p1);
    }
    gettimeofday(&end, NULL);
    printf("CPU time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    FDTD<float, nz, nx, ny, 4> fdtd;
    float *gpu_p0 = fdtd.mallocCube("p0");
    float *gpu_p1 = fdtd.mallocCube("p1");
    fdtd.transferCubeToGPU("p0", cpu_p0);
    fdtd.transferCubeToGPU("p1", cpu_p1);
    auto kernel = [=] __device__ (size_t z, size_t y, size_t x, float &output, float* zl, float *yl, float *xl, float scal)
    {
        output = scal*(xl[0]+
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
                       3.0*(zl[4]+zl[-4])
                       );
    };
    gettimeofday(&start, NULL);
    for(size_t t=0;t<20;t++)
    {
        if(t%2 == 0)
            fdtd.propagate("p0", "p1", true, kernel, 1.0/(6*8+1));
        else
            fdtd.propagate("p1", "p0", true, kernel, 1.0/(6*8+1));
    }
    gettimeofday(&end, NULL);
    printf("GPU time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    fdtd.transferCubeToCPU(cpu_p0, "p0");
    fdtd.transferCubeToCPU(cpu_p1, "p1");
    for(size_t i=0;i<nz;i++)
        for(size_t j=0;j<ny;j++)
            for(size_t k=0;k<nx;k++)
            {
                if(fabs(p0[GET(i,j,k,nz,ny,nx)]-cpu_p0[GET(i,j,k,nz,ny,nx)])>1e-5 || fabs(p1[GET(i,j,k,nz,ny,nx)]-cpu_p1[GET(i,j,k,nz,ny,nx)])>1e-5)
                    printf("[%lu][%lu][%lu]: %lf %lf, %lf %lf\n", i, j, k, p0[GET(i,j,k,nz,ny,nx)], cpu_p0[GET(i,j,k,nz,ny,nx)], p1[GET(i,j,k,nz,ny,nx)], cpu_p1[GET(i,j,k,nz,ny,nx)]);
            }
    cudaDeviceSynchronize();
}
