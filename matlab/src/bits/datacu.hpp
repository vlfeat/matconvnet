//
//  datacu.hpp
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__datacu__
#define __matconv__datacu__

#include "data.hpp"

#ifndef ENABLE_GPU
#error "datacu.hpp cannot be used without GPU support"
#endif

#include <cuda.h>
#include <cublas_v2.h>
#if __CUDA_ARCH__ >= 200
#define VL_CUDA_NUM_THREADS 1024
#else
#define VL_CUDA_NUM_THREADS 512
#endif

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace vl {
  class CudaHelper {
  public:
    CudaHelper() ;
    ~CudaHelper() ;

    cublasStatus_t getCuBLASHandle(cublasHandle_t* handle) ;
#if ENABLE_CUDNN
    bool isCudnnActive() const ;
    void setCudnnActive(bool active) ;
    cudnnStatus_t getCuDNNHandle(cudnnHandle_t* handle) ;
#endif

  private:
    cublasHandle_t cuBLASHandle ;
    bool isCuBLASInitialized ;

#if ENABLE_CUDNN
    bool cudnnActive ;
    cudnnHandle_t cuDNNHandle ;
    bool isCuDNNInitialized ;
#endif
  } ;
}
#endif /* defined(__matconv__datacu__) */
