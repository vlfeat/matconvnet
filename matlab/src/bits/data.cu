// @file data.cu
// @brief Basic data structures
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
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
    "unsupported feature error",
    "CUDA error",
    "cuDNN error",
    "cuBLAS error",
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
    setError(error, "getWorkspace") ;
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
 *                                                         TensorShape
 * ---------------------------------------------------------------- */

vl::TensorShape::TensorShape()
: numDimensions(0)
{ }

vl::TensorShape::TensorShape(TensorShape const & t)
: numDimensions(t.numDimensions)
{
  for (unsigned k = 0 ; k < numDimensions ; ++k) {
    dimensions[k] = t.dimensions[k] ;
  }
}

vl::TensorShape::TensorShape(size_t height, size_t width, size_t depth, size_t size)
: numDimensions(4)
{
  dimensions[0] = height ;
  dimensions[1] = width ;
  dimensions[2] = depth ;
  dimensions[3] = size ;
}

void vl::TensorShape::clear()
{
  numDimensions = 0 ;
}

void vl::TensorShape::setDimension(size_t num, size_t dimension)
{
  assert(num + 1 <= VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS) ;
  if (num + 1 > numDimensions) {
    size_t x = (getNumElements() > 0) ;
    for (size_t k = numDimensions ; k < num ; ++k) {
      dimensions[k] = x ;
    }
    numDimensions = num + 1 ;
  }
  dimensions[num] = dimension ;
}

size_t vl::TensorShape::getDimension(size_t num) const
{
  if (num + 1 > numDimensions) {
    return 1 ;
  }
  return dimensions[num] ;
}

size_t vl::TensorShape::getNumDimensions() const
{
  return numDimensions ;
}

size_t const * vl::TensorShape::getDimensions() const
{
  return dimensions ;
}

size_t vl::TensorShape::getNumElements() const
{
  if (numDimensions == 0) {
    return 0 ;
  }
  size_t n = 1 ;
  for (unsigned k = 0 ; k < numDimensions ; ++k) { n *= dimensions[k] ; }
  return n ;
}

size_t vl::TensorShape::getHeight() const { return getDimension(0) ; }
size_t vl::TensorShape::getWidth() const { return getDimension(1) ; }
size_t vl::TensorShape::getDepth() const { return getDimension(2) ; }
size_t vl::TensorShape::getSize() const { return getDimension(3) ; }

void vl::TensorShape::setHeight(size_t x) { setDimension(0,x) ; }
void vl::TensorShape::setWidth(size_t x) { setDimension(1,x) ; }
void vl::TensorShape::setDepth(size_t x) { setDimension(2,x) ; }
void vl::TensorShape::setSize(size_t x) { setDimension(3,x) ; }
bool vl::TensorShape::isEmpty() const { return getNumElements() == 0 ; }

bool vl::operator== (vl::TensorShape const & a, vl::TensorShape const & b)
{
  size_t n = a.getNumDimensions() ;
  if (b.getNumDimensions() != n) { return false ; }
  size_t const * adims = a.getDimensions() ;
  size_t const * bdims = b.getDimensions() ;
  for (unsigned k =0 ; k < a.getNumDimensions() ; ++k) {
    if (adims[k] != bdims[k]) { return false ; }
  }
  return true ;
}

void vl::TensorShape::reshape(size_t newNumDimensions)
{
  assert(newNumDimensions <= VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS) ;
  size_t n = getNumElements() ;
  if (newNumDimensions > 0) {
    setDimension(newNumDimensions - 1, 1) ;
    numDimensions = newNumDimensions ;
    size_t m = getNumElements() ;
    if (m) {
      dimensions[newNumDimensions - 1] *= (n / m) ;
    } else if (n == 0) {
      dimensions[newNumDimensions - 1] = 0  ;
    }
  } else {
    numDimensions = newNumDimensions ;
  }
}

void vl::TensorShape::reshape(TensorShape const & newShape)
{
  operator=(newShape) ;
}

/* -------------------------------------------------------------------
 *                                                              Tensor
 * ---------------------------------------------------------------- */

vl::Tensor::Tensor()
: TensorShape(), dataType(vlTypeFloat),
  deviceType(CPU), memory(NULL), memorySize(0)
{ }

vl::Tensor::Tensor(TensorShape const & shape, Type dataType,
                   Device deviceType, void * memory, size_t memorySize)
: TensorShape(shape),
dataType(dataType),
deviceType(deviceType),
memory(memory), memorySize(memorySize)
{ }

TensorShape vl::Tensor::getShape() const
{
  return TensorShape(*this) ;
}

vl::Type vl::Tensor::getDataType() const { return dataType ; }
void * vl::Tensor::getMemory() { return memory ; }
void vl::Tensor::setMemory(void * x) { memory = x ; }
vl::Device vl::Tensor::getDeviceType() const { return deviceType ; }
bool vl::Tensor::isNull() const { return memory == NULL ; }
vl::Tensor::operator bool() const { return !isNull() ; }


