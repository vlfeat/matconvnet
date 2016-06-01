// @file datacu.cu
// @brief Basic data structures (CUDA support)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef ENABLE_GPU
#error "datacu.cu cannot be compiled without GPU support"
#endif

#include "datacu.hpp"
#include "impl/blashelper.hpp"
#include <cublas_v2.h>

#ifndef NDEBUG
#include <iostream>
#endif

using namespace vl ;

/* -------------------------------------------------------------------
 * CudaHelper
 * ---------------------------------------------------------------- */

vl::CudaHelper::CudaHelper()
: isCublasInitialized(false)
#if ENABLE_CUDNN
, isCudnnInitialized(false), cudnnEnabled(true)
#endif
{
#if ENABLE_CUDNN
  resetCudnnConvolutionSettings() ;
#endif
}

vl::CudaHelper::~CudaHelper()
{
  clear() ;
}

void
vl::CudaHelper::clear()
{
  clearCublas() ;
#ifdef ENABLE_CUDNN
  clearCudnn() ;
#endif
}

void
vl::CudaHelper::invalidateGpu()
{
#ifndef NDEBUG
  std::cout<<"CudaHelper::invalidateGpu()"<<std::endl ;
#endif
  isCublasInitialized = false ;
#ifdef ENABLE_CUDNN
  isCudnnInitialized = false ;
#endif
}

/* -------------------------------------------------------------------
 * getCublasHandle
 * ---------------------------------------------------------------- */

cublasStatus_t
vl::CudaHelper::getCublasHandle(cublasHandle_t* handle)
{
  if (!isCublasInitialized) {
    clearCublas() ;
    cublasStatus_t stat = cublasCreate(&cublasHandle) ;
    if (stat != CUBLAS_STATUS_SUCCESS) { return stat ; }
    isCublasInitialized = true ;
  }
  *handle = cublasHandle ;
  return CUBLAS_STATUS_SUCCESS ;
}

void
vl::CudaHelper::clearCublas()
{
  if (!isCublasInitialized) { return ; }
  cublasDestroy(cublasHandle) ;
  isCublasInitialized = false ;
}

/* -------------------------------------------------------------------
 * getCudnnHandle
 * ---------------------------------------------------------------- */

#if ENABLE_CUDNN
cudnnStatus_t
vl::CudaHelper::getCudnnHandle(cudnnHandle_t* handle)
{
  if (!isCudnnInitialized) {
    clearCudnn() ;
    cudnnStatus_t stat = cudnnCreate(&cudnnHandle) ;
    if (stat != CUDNN_STATUS_SUCCESS) { return stat ; }
    isCudnnInitialized = true ;
  }
  *handle = cudnnHandle ;
  return CUDNN_STATUS_SUCCESS ;
}

void
vl::CudaHelper::clearCudnn()
{
  if (!isCudnnInitialized) { return ; }
  cudnnDestroy(cudnnHandle) ;
  isCudnnInitialized = false ;
}

bool
vl::CudaHelper::getCudnnEnabled() const
{
  return cudnnEnabled ;
}

void
vl::CudaHelper::setCudnnEnabled(bool active)
{
  cudnnEnabled = active ;
}

/* -------------------------------------------------------------------
 * cuDNN parameters
 * ---------------------------------------------------------------- */

void
vl::CudaHelper::resetCudnnConvolutionSettings()
{
  cudnnConvolutionFwdSpecificAlgo = false ;
  cudnnConvolutionFwdPreference = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT ;
  cudnnConvolutionFwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM ;
  cudnnConvolutionFwdWorkSpaceLimit = 512 * 1024 * 1024 ; // 512MB
  cudnnConvolutionFwdWorkSpaceUsed = 0 ;

  cudnnConvolutionBwdFilterSpecificAlgo = false ;
  cudnnConvolutionBwdFilterPreference = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT ;
  cudnnConvolutionBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 ;
  cudnnConvolutionBwdFilterWorkSpaceLimit = 512 * 1024 * 1024 ; // 512MB
  cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;

  cudnnConvolutionBwdDataSpecificAlgo = false ;
  cudnnConvolutionBwdDataPreference = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT ;
  cudnnConvolutionBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 ;
  cudnnConvolutionBwdDataWorkSpaceLimit = 512 * 1024 * 1024 ; // 512MB
  cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;
}

void
vl::CudaHelper::setCudnnConvolutionFwdPreference(cudnnConvolutionFwdPreference_t x,
                                                 size_t workSpaceLimit)
{
  cudnnConvolutionFwdSpecificAlgo = false ;
  cudnnConvolutionFwdPreference = x ;
  cudnnConvolutionFwdWorkSpaceLimit = workSpaceLimit ;
}

void
vl::CudaHelper::setCudnnConvolutionFwdAlgo(cudnnConvolutionFwdAlgo_t x)
{
  cudnnConvolutionFwdSpecificAlgo = true ;
  cudnnConvolutionFwdAlgo = x ;
}

size_t
vl::CudaHelper::getCudnnConvolutionFwdWorkSpaceUsed() const
{
  return cudnnConvolutionFwdWorkSpaceUsed ;
}

void
vl::CudaHelper::setCudnnConvolutionBwdFilterPreference(cudnnConvolutionBwdFilterPreference_t x,
                                                       size_t workSpaceLimit)
{
  cudnnConvolutionBwdFilterSpecificAlgo = false ;
  cudnnConvolutionBwdFilterPreference = x ;
  cudnnConvolutionBwdFilterWorkSpaceLimit = workSpaceLimit ;
}

void
vl::CudaHelper::setCudnnConvolutionBwdFilterAlgo(cudnnConvolutionBwdFilterAlgo_t x)

{
  cudnnConvolutionBwdFilterSpecificAlgo = true ;
  cudnnConvolutionBwdFilterAlgo = x ;
}

size_t
vl::CudaHelper::getCudnnConvolutionBwdFilterWorkSpaceUsed() const
{
  return cudnnConvolutionBwdFilterWorkSpaceUsed ;
}

void
vl::CudaHelper::setCudnnConvolutionBwdDataPreference(cudnnConvolutionBwdDataPreference_t x,
                                                     size_t workSpaceLimit)
{
  cudnnConvolutionBwdDataSpecificAlgo = false ;
  cudnnConvolutionBwdDataPreference = x ;
  cudnnConvolutionBwdDataWorkSpaceLimit = workSpaceLimit ;
}

void
vl::CudaHelper::setCudnnConvolutionBwdDataAlgo(cudnnConvolutionBwdDataAlgo_t x)
{
  cudnnConvolutionBwdDataSpecificAlgo = true ;
  cudnnConvolutionBwdDataAlgo = x ;
}

size_t
vl::CudaHelper::getCudnnConvolutionBwdDataWorkSpaceUsed() const
{
  return cudnnConvolutionBwdDataWorkSpaceUsed ;
}
#endif

/* -------------------------------------------------------------------
 * CuBLAS Errors
 * ---------------------------------------------------------------- */

static const char *
getCublasErrorMessageFromStatus(cublasStatus_t status)
{
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "CuBLAS unknown status" ;
}

vl::Error
vl::CudaHelper::catchCublasError(cublasStatus_t status, char const * description)
{
  /* if there is no CuBLAS error, do not do anything */
  if (status == CUBLAS_STATUS_SUCCESS) { return vl::vlSuccess ; }

  /* if there is a CuBLAS error, store it */
  lastCublasError = status ;
  std::string message = getCublasErrorMessageFromStatus(status) ;
  if (description) {
    message = std::string(description) + " (" + message + ")" ;
  }
  lastCublasErrorMessage = message ;
  return vl::vlErrorCublas ;
}

cublasStatus_t
vl::CudaHelper::getLastCublasError() const
{
  return lastCublasError;
}

std::string const&
vl::CudaHelper::getLastCublasErrorMessage() const
{
  return lastCublasErrorMessage ;
}

/* -------------------------------------------------------------------
 * CuDNN Errors
 * ---------------------------------------------------------------- */

#if ENABLE_CUDNN
vl::Error
vl::CudaHelper::catchCudnnError(cudnnStatus_t status, char const* description)
{
  /* if there is no CuDNN error, do not do anything */
  if (status == CUDNN_STATUS_SUCCESS) { return vl::vlSuccess ; }

  /* if there is a CuDNN error, store it */
  lastCudnnError = status ;
  std::string message = cudnnGetErrorString(status) ;
  if (description) {
    message = std::string(description) + " (" + message + ")" ;
  }
  lastCudnnErrorMessage = message ;
  return vl::vlErrorCudnn ;
}

cudnnStatus_t
vl::CudaHelper::getLastCudnnError() const
{
  return lastCudnnError;
}

std::string const&
vl::CudaHelper::getLastCudnnErrorMessage() const
{
  return lastCudnnErrorMessage ;
}
#endif

/* -------------------------------------------------------------------
 * Cuda Errors
 * ---------------------------------------------------------------- */

vl::Error
vl::CudaHelper::catchCudaError(char const* description)
{
  /* if there is no Cuda error, do not do anything */
  cudaError_t error = cudaPeekAtLastError() ;
  if (error == cudaSuccess) { return vl::vlSuccess ; }

  /* if there is a Cuda error, eat it and store it */
  lastCudaError = cudaGetLastError() ;
  std::string message = cudaGetErrorString(lastCudaError) ;
  if (description) {
    message = std::string(description) + ": " + message ;
  }
  lastCudaErrorMessage = message ;
  return vl::vlErrorCuda ;
}

cudaError_t
vl::CudaHelper::getLastCudaError() const
{
  return lastCudaError ;
}

std::string const&
vl::CudaHelper::getLastCudaErrorMessage() const
{
  return lastCudaErrorMessage ;
}



