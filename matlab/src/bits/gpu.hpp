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

#ifndef VL_GPU_H
#define VL_GPU_H

#include <iostream>

#if __CUDA_ARCH__ >= 200
#define VL_CUDA_NUM_THREADS 1024
#else
#define VL_CUDA_NUM_THREADS 512
#endif

inline int divideUpwards(int a, int b)
{
  return (a + b - 1) / b ;
}

#endif
