cmake_minimum_required(VERSION 2.8)
project(FDTD)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -arch=sm_61;-gencode arch=compute_70,code=sm_70;-gencode arch=compute_60,code=sm_60;-O3;--expt-extended-lambda;-g -lineinfo;-Xptxas -v;-w)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(SEND_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

CHECK_CXX_COMPILER_FLAG("-Ofast -fopenmp" COMPILER_SUPPORTS_OFAST)
if(COMPILER_SUPPORTS_OFAST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

enable_language(CXX)
#file(GLOB SOURCES *.cpp)
#file(GLOB CUDA_SOURCES *.cu)

cuda_add_executable(fdtd main.cu)
cuda_add_executable(fdtd4 fdtd4.cu)
cuda_add_executable(fdtd1 fdtd1.cu)
