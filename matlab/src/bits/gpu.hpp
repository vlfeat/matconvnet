/** @file    gpu.h
 ** @brief   GPU helper functions.
 ** @author  Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef matconv_caffe_scraps_h
#define matconv_caffe_scraps_h

#include <iostream>

#define CUDA_POST_KERNEL_CHECK \
if (cudaSuccess != cudaPeekAtLastError()) \
std::cout << "[Caffe]: Cuda kernel failed. Error: " \
<< cudaGetErrorString(cudaPeekAtLastError()) << std::endl

// We will use 1024 threads per block, which requires cuda sm_2x or above.
#if __CUDA_ARCH__ >= 200
const int CAFFE_CUDA_NUM_THREADS = 1024;
#define VL_CUDA_NUM_THREADS 1024
#else
const int CAFFE_CUDA_NUM_THREADS = 512;
#define VL_CUDA_NUM_THREADS 512
#endif

inline int CAFFE_GET_BLOCKS(const int N)
{
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

inline int divideUpwards(int a, int b)
{
  return (a + b - 1) / b ;
}

#endif
