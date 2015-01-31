//
//  datacu.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "datacu.hpp"

using namespace vl ;

#ifndef ENABLE_GPU
#error "datacu.cpp cannot be compiled without GPU support"
#endif

/* -------------------------------------------------------------------
 * CudaHelper
 * ---------------------------------------------------------------- */

vl::CudaHelper::CudaHelper()
: isCuBLASInitialized(false)
#if ENABLE_CUDNN
, isCuDNNInitialized(false), cudnnActive(true)
#endif
{ }

vl::CudaHelper::~CudaHelper()
{
  if (isCuBLASInitialized) {
    cublasDestroy(cuBLASHandle) ;
    isCuBLASInitialized = false ;
  }
#ifdef ENABLE_CUDNN
  if (isCuDNNInitialized) {
    cudnnDestroy(cuDNNHandle) ;
    isCuDNNInitialized = false ;
  }
#endif
}

/* -------------------------------------------------------------------
 * getCuBLASHandle
 * ---------------------------------------------------------------- */

cublasStatus_t vl::CudaHelper::getCuBLASHandle(cublasHandle_t* handle)
{
  if (!isCuBLASInitialized) {
    cublasStatus_t stat = cublasCreate(&cuBLASHandle) ;
    if (stat != CUBLAS_STATUS_SUCCESS) { return stat ; }
    isCuBLASInitialized = true ;
  }
  *handle = cuBLASHandle ;
  return CUBLAS_STATUS_SUCCESS ;
}

/* -------------------------------------------------------------------
 * getCuDNNHandle
 * ---------------------------------------------------------------- */

#if ENABLE_CUDNN
cudnnStatus_t vl::CudaHelper::getCuDNNHandle(cudnnHandle_t* handle)
{
  if (!isCuDNNInitialized) {
    cudnnStatus_t stat = cudnnCreate(&cuDNNHandle) ;
    if (stat != CUDNN_STATUS_SUCCESS) { return stat ; }
    isCuDNNInitialized = true ;
  }
  *handle = cuDNNHandle ;
  return CUDNN_STATUS_SUCCESS ;
}

bool vl::CudaHelper::isCudnnActive() const
{
  return cudnnActive ;
}

void vl::CudaHelper::setCudnnActive(bool active)
{
  cudnnActive = active ;
}
#endif



