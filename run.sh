#!/bin/bash
export MV2_GPUDIRECT_GDRCOPY_LIB=/home/jiguanglizipao/.local/lib64/libgdrapi.so
export MV2_USE_CUDA=1
export MV2_CUDA_ENABLE_MANAGED=1
export MV2_CUDA_MANAGED_IPC=1
export MV2_USE_CORE_DIRECT=3
export MV2_ENABLE_AFFINITY=0
export MV2_SMP_USE_CMA=0

if [ -z $GPU ]
then
    GPU=0
fi

if [ -z $CPU ]
then
    CPU=$GPU
fi

export CUDA_VISIBLE_DEVICES=$GPU
np=$1
shift 1
mpirun_rsh -export -ssh -np $np $@
