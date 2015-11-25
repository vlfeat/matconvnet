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

static int
getTypeSize(Type dataType)
{
  switch (dataType) {
    case vlTypeChar : return sizeof(char) ;
    case vlTypeFloat : return sizeof(float) ;
    case vlTypeDouble : return sizeof(double) ;
    default: abort() ;
  }
  return 0 ;
}

/* -------------------------------------------------------------------
 * Buffer
 * ---------------------------------------------------------------- */

vl::impl::Buffer::Buffer()
:
deviceType(vl::CPU), dataType(vlTypeChar),
size(0), memory(NULL), numReallocations(0)
{ }

void*
vl::impl::Buffer::getMemory()
{
  return memory ;
}

int
vl::impl::Buffer::getNumReallocations() const
{
  return numReallocations ;
}

vl::Error
vl::impl::Buffer::init(Device deviceType_, Type dataType_, size_t size_)
{
  bool ok =
  (deviceType == deviceType_) &
  (dataType == dataType_) &
  (size >= size_) ;
  if (ok) { return vl::vlSuccess ; }
  clear() ;
  void * memory_ = NULL ;
  size_t sizeInBytes = getTypeSize(dataType_) * size_ ;
  switch (deviceType_) {
    case vl::CPU:
      memory_ = malloc(sizeInBytes) ;
      if (memory_ == NULL) { return vl::vlErrorOutOfMemory ; }
      break ;
    case vl::GPU:
#if ENABLE_GPU
      cudaError_t error = cudaMalloc(&memory_, sizeInBytes) ;
      if (error != cudaSuccess) { return vl::vlErrorOutOfMemory ; }
      break ;
#else
      abort() ;
#endif
  }
  deviceType = deviceType_ ;
  dataType = dataType_ ;
  size = size_ ;
  memory = memory_ ;
  numReallocations ++ ;
  return vl::vlSuccess ;
}

void
vl::impl::Buffer::clear()
{
  if (memory != NULL) {
    switch (deviceType) {
      case vl::CPU:
        free(memory) ;
        break ;
      case vl::GPU:
#if ENABLE_GPU
        cudaFree(memory) ;
        break ;
#else
        abort() ;
#endif
    }
  }
  deviceType = vl::CPU ;
  dataType= vlTypeChar ;
  size = 0 ;
  memory = NULL ;
}

void
vl::impl::Buffer::invalidateGpu()
{
  if (deviceType == vl::GPU) {
    memory = NULL ;
    clear() ;
  }
}

/* -------------------------------------------------------------------
 * Context
 * ---------------------------------------------------------------- */

vl::Context::Context()
:
lastError(vl::vlSuccess), lastErrorMessage(), cudaHelper(NULL)
{ }

vl::CudaHelper &
vl::Context::getCudaHelper()
{
#ifdef ENABLE_GPU
  if (!cudaHelper) {
    cudaHelper = new CudaHelper() ;
  }
#else
  abort() ;
#endif
  return *cudaHelper ;
}

void vl::Context::clear()
{
#ifndef NDEBUG
  std::cout<<"Context::clear()"<<std::endl ;
#endif
  clearWorkspace(CPU) ;
  clearAllOnes(CPU) ;
#if ENABLE_GPU
  clearWorkspace(GPU) ;
  clearAllOnes(GPU) ;
  if (cudaHelper) {
    delete cudaHelper ;
    cudaHelper = NULL ;
  }
#endif
}

void
vl::Context::invalidateGpu()
{
#if ENABLE_GPU
  workspace[vl::GPU].invalidateGpu() ;
  allOnes[vl::GPU].invalidateGpu() ;
  getCudaHelper().invalidateGpu() ;
#endif
}

vl::Context::~Context()
{
  clear() ;
#ifndef NDEBUG
  std::cout<<"Context::~Context()"<<std::endl ;
#endif
}

/* -------------------------------------------------------------------
 * Context errors
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
 * Context workspace
 * ---------------------------------------------------------------- */

void *
vl::Context::getWorkspace(Device deviceType, size_t size)
{
  vl::Error error = workspace[deviceType].init(deviceType, vlTypeChar, size) ;
  if (error != vlSuccess) {
    setError(error, "getWorkspace: ") ;
    return NULL ;
  }
  return workspace[deviceType].getMemory() ;
}

void
vl::Context::clearWorkspace(Device deviceType)
{
  workspace[deviceType].clear() ;
}

/* -------------------------------------------------------------------
 * Context allOnes
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
vl::Context::getAllOnes(Device deviceType, Type dataType, size_t size)
{
  int n = allOnes[deviceType].getNumReallocations() ;
  void * data = NULL ;

  // make sure that there is enough space for the buffer
  vl::Error error = allOnes[deviceType].init(deviceType, dataType, size) ;
  if (error != vlSuccess) { goto done ; }
  data = allOnes[deviceType].getMemory() ;

  // detect if a new buffer has been allocated and if so initialise it
  if (n < allOnes[deviceType].getNumReallocations()) {
    switch (deviceType) {
      case vl::CPU:
        for (int i = 0 ; i < size ; ++i) {
          if (dataType == vlTypeFloat) {
            ((float*)data)[i] = 1.0f ;
          } else {
            ((double*)data)[i] = 1.0 ;
          }
        }
        break ;

      case GPU:
#if ENABLE_GPU
        if (dataType == vlTypeFloat) {
          setToOnes<float>
          <<<divideUpwards(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
          ((float*)data, size) ;
        } else {
          setToOnes<double>
          <<<divideUpwards(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
          ((double*)data, size) ;
        }
        error = getCudaHelper().catchCudaError() ;
        break ;
#else
        abort() ;
        return NULL ;
#endif
    }
  }
done:
  if (setError(error, "getAllOnes: ") == vl::vlSuccess) {
    return data ;
  } else {
    return NULL ;
  }
}

void
vl::Context::clearAllOnes(Device deviceType)
{
  allOnes[deviceType].clear() ;
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
void vl::TensorGeometry::setHeight(index_t x) { height = x ; }
void vl::TensorGeometry::setWidth(index_t x) { width = x ; }
void vl::TensorGeometry::setDepth(index_t x) { depth = x ; }
void vl::TensorGeometry::setSize(index_t x) { size = x ; }
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
void vl::Tensor::setMemory(float * x) { memory = x ; }
vl::Device vl::Tensor::getMemoryType() const { return memoryType ; }
bool vl::Tensor::isNull() const { return memory == NULL ; }
vl::Tensor::operator bool() const { return !isNull() ; }

