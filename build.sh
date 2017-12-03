#!/bin/sh
rm -rf build
mkdir build
cd build
export CXX=g++
cmake -DCUDA_HOST_COMPILER=`which $CXX` ..
make -j
