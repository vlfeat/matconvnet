// @file data.cu
// @brief Basic data structures
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "data.hpp"
#include <cassert>
#include <cstdlib>

#ifndef NDEBUG
#include <iostream>
#endif

#if ENABLE_GPU
#include "datacu.hpp"
#endif

using namespace vl ;

/* -------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

const char *
vl::getErrorMessage(Error error)
{
  static char const * messages[] = {
    "success",
    "unsupported error",
    "cuda error",
    "cudnn error",
    "cublas error",
    "out of memory error",
    "out of GPU memory error",
    "unknown error"
  } ;
  if (error < vlSuccess || error > vlErrorUnknown) {
    error = vlErrorUnknown ;
  }
  return messages[error] ;
}

/* -------------------------------------------------------------------
 * Context
 * ---------------------------------------------------------------- */

vl::Context::Context()
:
lastError(vl::vlSuccess), lastErrorMessage(),
cpuWorkspace(0), cpuWorkspaceSize(0),
gpuWorkspace(0), gpuWorkspaceSize(0),
cudaHelper(NULL)
{ }

vl::CudaHelper &
vl::Context::getCudaHelper()
{
#ifdef ENABLE_GPU
  if (!cudaHelper) {
    cudaHelper = new CudaHelper() ;
  }
  return *cudaHelper ;
#else
  abort() ;
#endif
}

void vl::Context::reset()
{
  resetLastError() ;
  clearWorkspace(CPU) ;
  clearAllOnes(CPU) ;
#if ENABLE_GPU
  clearWorkspace(GPU) ;
  clearAllOnes(GPU) ;
  if (cudaHelper) {
    delete cudaHelper ;
    cudaHelper = 0 ;
  }
#endif
#ifndef NDEBUG
  std::cout<<"Context::reset()"<<std::endl ;
#endif
}

vl::Context::~Context()
{
  reset() ;
#ifndef NDEBUG
  std::cout<<"Context::~Context()"<<std::endl ;
#endif
}

/* -------------------------------------------------------------------
 * Errors
 * ---------------------------------------------------------------- */

void
vl::Context::resetLastError()
{
  lastError = vl::vlSuccess ;
  lastErrorMessage = std::string() ;
}

vl::Error
vl::Context::passError(vl::Error error, char const* description)
{
  if (error != vl::vlSuccess) {
    if (description) {
      lastErrorMessage = std::string(description) + ": " + lastErrorMessage ;
    }
  }
  return error ;
}

vl::Error
vl::Context::setError(vl::Error error, char const* description)
{
  if (error != vl::vlSuccess ) {
    lastError = error ;
    std::string message = getErrorMessage(error) ;
    if (description) {
      message = std::string(description) + " [" + message + "]" ;
    }
#if ENABLE_GPU
    if (error == vl::vlErrorCuda) {
      std::string cudaMessage = getCudaHelper().getLastCudaErrorMessage() ;
      if (cudaMessage.length() > 0) {
        message += " [cuda: " + cudaMessage + "]" ;
      }
    }
    if (error == vl::vlErrorCublas) {
      std::string cublasMessage = getCudaHelper().getLastCublasErrorMessage() ;
      if (cublasMessage.length() > 0) {
        message += " [cublas:" + cublasMessage + "]" ;
      }
    }
#endif
#if ENABLE_CUDNN
    if (error == vl::vlErrorCudnn) {
      std::string cudnnMessage = getCudaHelper().getLastCudnnErrorMessage() ;
      if (cudnnMessage.length() > 0) {
        message += " [cudnn: " + cudnnMessage + "]" ;
      }
    }
#endif
    lastErrorMessage = message ;
  }
  return error ;
}

vl::Error
vl::Context::getLastError() const
{
  return lastError ;
}

std::string const&
vl::Context::getLastErrorMessage() const
{
  return lastErrorMessage ;
}

/* -------------------------------------------------------------------
 * getWorkspace
 * ---------------------------------------------------------------- */

void *
vl::Context::getWorkspace(Device device, size_t size)
{
  switch (device) {
    case CPU:
      if (cpuWorkspaceSize < size) {
        clearWorkspace(CPU) ;
        cpuWorkspace = malloc(size) ;
        if (cpuWorkspace == NULL) {
          setError(vl::vlErrorOutOfMemory, "getWorkspace") ;
          return NULL ;
        }
        cpuWorkspaceSize = size ;
      }
      return cpuWorkspace ;

    case GPU:
#if ENABLE_GPU
      if (gpuWorkspaceSize < size) {
        clearWorkspace(GPU) ;
        cudaError_t status = cudaMalloc(&gpuWorkspace, size) ;
        if (status != cudaSuccess) {
          setError(getCudaHelper().catchCudaError(), "getWorkspace") ;
          return NULL ;
        }
        gpuWorkspaceSize = size ;
      }
      return gpuWorkspace ;
#else
      assert(false) ;
      return NULL ;
#endif
    default:
      assert(false) ;
      return NULL ;
  }
}

/* -------------------------------------------------------------------
 * clearWorkspace
 * ---------------------------------------------------------------- */

void
vl::Context::clearWorkspace(Device device)
{
  switch (device) {
    case CPU:
      if (cpuWorkspaceSize > 0) {
        free(cpuWorkspace) ;
        cpuWorkspace = NULL ;
        cpuWorkspaceSize = 0 ;
      }
      break ;
    case GPU:
#if ENABLE_GPU
      if (gpuWorkspaceSize > 0) {
        cudaFree(gpuWorkspace) ;
        gpuWorkspace = NULL ;
        gpuWorkspaceSize = 0 ;
      }
      break ;
#else
      assert(false) ;
#endif
  }
}

/* -------------------------------------------------------------------
 * getAllOnes
 * ---------------------------------------------------------------- */

#if ENABLE_GPU
template<typename type> __global__ void
setToOnes (type * data, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < size) data[index] = type(1.0) ;
}
#endif

void *
vl::Context::getAllOnes(Device device, Type type, size_t size)
{
  int typeSize = (type == vlTypeFloat) ? sizeof(float) : sizeof(double) ;

  switch (device) {
    default:
      assert(false) ;
      lastError = vl::vlErrorUnknown ;
      return NULL ;

    case CPU:
      if (cpuAllOnesSize < size || cpuAllOnesType != type) {
        clearAllOnes(CPU) ;
        cpuAllOnes = malloc(size * typeSize) ;
        if (cpuAllOnes == NULL) {
          setError(vlErrorOutOfMemory, "getAllOnes") ;
          return NULL ;
        }
        cpuAllOnesType = type ;
        cpuAllOnesSize = size ;
        for (int i = 0 ; i < size ; ++i) {
          if (type == vlTypeFloat) {
            ((float*)cpuAllOnes)[i] = 1.0f ;
          } else {
            ((double*)cpuAllOnes)[i] = 1.0 ;
          }
        }
      }
      return cpuAllOnes ;

    case GPU:
#if ENABLE_GPU
      if (gpuAllOnesSize < size || gpuAllOnesType != type) {
        cudaError_t status ;
        clearAllOnes(GPU) ;
        status = cudaMalloc(&gpuAllOnes, size * typeSize) ;
        if (status != cudaSuccess) {
          setError(getCudaHelper().catchCudaError(), "getAllOnes") ;
          return NULL ;
        }
        gpuAllOnesType = type ;
        gpuAllOnesSize = size ;
        if (type == vlTypeFloat) {
          setToOnes<float>
          <<<divideUpwards(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
          ((float*)gpuAllOnes, size) ;
        } else {
          setToOnes<double>
          <<<divideUpwards(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
          ((double*)gpuAllOnes, size) ;
        }
        setError(getCudaHelper().catchCudaError(), "getAllOnes") ;
        return NULL ;
      }
      return gpuAllOnes ;
#else
      assert(false) ;
      lastError = vl::vlErrorUnknown ;
      return NULL ;
#endif
  }
}

/* -------------------------------------------------------------------
 * clearAllOnes
 * ---------------------------------------------------------------- */

void
vl::Context::clearAllOnes(Device device)
{
  switch (device) {
    case CPU:
      if (cpuAllOnesSize > 0) {
        free(cpuAllOnes) ;
        cpuAllOnes = NULL ;
        cpuAllOnesSize = 0 ;
      }
      break ;
    case GPU:
#if ENABLE_GPU
      if (gpuAllOnesSize > 0) {
        cudaFree(gpuAllOnes) ;
        gpuAllOnes = NULL ;
        gpuAllOnesSize = 0 ;
      }
      break ;
#else
      assert(false) ;
#endif
    default:
      assert(false) ;
      return ;
  }
}

/* -------------------------------------------------------------------
 * TensorGeometry
 * ---------------------------------------------------------------- */

vl::TensorGeometry::TensorGeometry()
: height(0), width(0), depth(0), size(0)
{ }

vl::TensorGeometry::TensorGeometry(index_t height, index_t width, index_t depth, index_t size)
: height(height), width(width), depth(depth), size(size)
{ }

vl::index_t vl::TensorGeometry::getHeight() const { return height ; }
vl::index_t vl::TensorGeometry::getWidth() const { return width ; }
vl::index_t vl::TensorGeometry::getDepth() const { return depth ; }
vl::index_t vl::TensorGeometry::getSize() const { return size ; }
vl::index_t vl::TensorGeometry::getNumElements() const { return height*width*depth*size ; }
bool vl::TensorGeometry::isEmpty() const { return getNumElements() == 0 ; }

/* -------------------------------------------------------------------
 * Tensor
 * ---------------------------------------------------------------- */

vl::Tensor::Tensor()
: TensorGeometry(), memory(NULL), memorySize(0), memoryType(CPU)
{ }

vl::Tensor::Tensor(float * memory, size_t memorySize, Device memoryType,
                   vl::TensorGeometry const & geom)
: TensorGeometry(geom),
memory(memory), memorySize(memorySize), memoryType(memoryType)
{ }

TensorGeometry vl::Tensor::getGeometry() const
{
  return TensorGeometry(*this) ;
}

float * vl::Tensor::getMemory() { return memory ; }
vl::Device vl::Tensor::getMemoryType() const { return memoryType ; }
bool vl::Tensor::isNull() const { return memory == NULL ; }
vl::Tensor::operator bool() const { return !isNull() ; }

