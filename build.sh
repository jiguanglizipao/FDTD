#!/bin/sh
rm -rf build
mkdir build
cd build
export CXX=mpicxx
cmake -DCUDA_HOST_COMPILER=`which $CXX` ..
make -j
