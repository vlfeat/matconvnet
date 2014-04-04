//
//  caffe-scraps.h
//  matconv
//
//  Created by Andrea Vedaldi on 13/02/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#ifndef matconv_caffe_scraps_h
#define matconv_caffe_scraps_h

#include <iostream>

#define FATAL std::cout
#define LOG(x) x

#define CUDA_POST_KERNEL_CHECK \
if (cudaSuccess != cudaPeekAtLastError()) \
LOG(FATAL) << "[Caffe]: Cuda kernel failed. Error: " \
<< cudaGetErrorString(cudaPeekAtLastError()) << std::endl

// We will use 1024 threads per block, which requires cuda sm_2x or above.
#if __CUDA_ARCH__ >= 200
const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

inline int CAFFE_GET_BLOCKS(const int N)
{
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

#endif
