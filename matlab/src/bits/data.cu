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
vl::getErrorMessage(ErrorCode error)
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
  if (error < VLE_Success || error > VLE_Unknown) {
    error = VLE_Unknown ;
  }
  return messages[error] ;
}

static int
getTypeSize(DataType dataType)
{
  switch (dataType) {
    case VLDT_Char : return sizeof(char) ;
    case VLDT_Float : return sizeof(float) ;
    case VLDT_Double : return sizeof(double) ;
    default: abort() ;
  }
  return 0 ;
}

/* -------------------------------------------------------------------
 * Buffer
 * ---------------------------------------------------------------- */

vl::impl::Buffer::Buffer()
:
deviceType(vl::VLDT_CPU), dataType(VLDT_Char),
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

vl::ErrorCode
vl::impl::Buffer::init(DeviceType deviceType_, DataType dataType_, size_t size_)
{
  bool ok =
  (deviceType == deviceType_) &
  (dataType == dataType_) &
  (size >= size_) ;
  if (ok) { return vl::VLE_Success ; }
  clear() ;
  void * memory_ = NULL ;
  size_t sizeInBytes = getTypeSize(dataType_) * size_ ;
  switch (deviceType_) {
    case vl::VLDT_CPU:
      memory_ = malloc(sizeInBytes) ;
      if (memory_ == NULL) { return vl::VLE_OutOfMemory ; }
      break ;
    case vl::VLDT_GPU:
#if ENABLE_GPU
      cudaError_t error = cudaMalloc(&memory_, sizeInBytes) ;
      if (error != cudaSuccess) { return vl::VLE_OutOfMemory ; }
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
  return vl::VLE_Success ;
}

void
vl::impl::Buffer::clear()
{
  if (memory != NULL) {
    switch (deviceType) {
      case vl::VLDT_CPU:
        free(memory) ;
        break ;
      case vl::VLDT_GPU:
#if ENABLE_GPU
        cudaFree(memory) ;
        break ;
#else
        abort() ;
#endif
    }
  }
  deviceType = vl::VLDT_CPU ;
  dataType= VLDT_Char ;
  size = 0 ;
  memory = NULL ;
}

void
vl::impl::Buffer::invalidateGpu()
{
  if (deviceType == vl::VLDT_GPU) {
    memory = NULL ;
    clear() ;
  }
}

/* -------------------------------------------------------------------
 * Context
 * ---------------------------------------------------------------- */

vl::Context::Context()
:
lastError(vl::VLE_Success), lastErrorMessage(), cudaHelper(NULL)
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
  clearWorkspace(VLDT_CPU) ;
  clearAllOnes(VLDT_CPU) ;
#if ENABLE_GPU
  clearWorkspace(VLDT_GPU) ;
  clearAllOnes(VLDT_GPU) ;
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
  workspace[vl::VLDT_GPU].invalidateGpu() ;
  allOnes[vl::VLDT_GPU].invalidateGpu() ;
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
  lastError = vl::VLE_Success ;
  lastErrorMessage = std::string() ;
}

vl::ErrorCode
vl::Context::passError(vl::ErrorCode error, char const* description)
{
  if (error != vl::VLE_Success) {
    if (description) {
      lastErrorMessage = std::string(description) + ": " + lastErrorMessage ;
    }
  }
  return error ;
}

vl::ErrorCode
vl::Context::setError(vl::ErrorCode error, char const* description)
{
  if (error != vl::VLE_Success ) {
    lastError = error ;
    std::string message = getErrorMessage(error) ;
    if (description) {
      message = std::string(description) + " [" + message + "]" ;
    }
#if ENABLE_GPU
    if (error == vl::VLE_Cuda) {
      std::string cudaMessage = getCudaHelper().getLastCudaErrorMessage() ;
      if (cudaMessage.length() > 0) {
        message += " [cuda: " + cudaMessage + "]" ;
      }
    }
    if (error == vl::VLE_Cublas) {
      std::string cublasMessage = getCudaHelper().getLastCublasErrorMessage() ;
      if (cublasMessage.length() > 0) {
        message += " [cublas:" + cublasMessage + "]" ;
      }
    }
#endif
#if ENABLE_CUDNN
    if (error == vl::VLE_Cudnn) {
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

vl::ErrorCode
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
vl::Context::getWorkspace(DeviceType deviceType, size_t size)
{
  vl::ErrorCode error = workspace[deviceType].init(deviceType, VLDT_Char, size) ;
  if (error != VLE_Success) {
    setError(error, "getWorkspace") ;
    return NULL ;
  }
  return workspace[deviceType].getMemory() ;
}

void
vl::Context::clearWorkspace(DeviceType deviceType)
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
vl::Context::getAllOnes(DeviceType deviceType, DataType dataType, size_t size)
{
  int n = allOnes[deviceType].getNumReallocations() ;
  void * data = NULL ;

  // make sure that there is enough space for the buffer
  vl::ErrorCode error = allOnes[deviceType].init(deviceType, dataType, size) ;
  if (error != VLE_Success) { goto done ; }
  data = allOnes[deviceType].getMemory() ;

  // detect if a new buffer has been allocated and if so initialise it
  if (n < allOnes[deviceType].getNumReallocations()) {
    switch (deviceType) {
      case vl::VLDT_CPU:
        for (int i = 0 ; i < size ; ++i) {
          if (dataType == VLDT_Float) {
            ((float*)data)[i] = 1.0f ;
          } else {
            ((double*)data)[i] = 1.0 ;
          }
        }
        break ;

      case vl::VLDT_GPU:
#if ENABLE_GPU
        if (dataType == VLDT_Float) {
          setToOnes<float>
          <<<divideAndRoundUp(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
          ((float*)data, size) ;
        } else {
          setToOnes<double>
          <<<divideAndRoundUp(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
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
  if (setError(error, "getAllOnes: ") == vl::VLE_Success) {
    return data ;
  } else {
    return NULL ;
  }
}

void
vl::Context::clearAllOnes(DeviceType deviceType)
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

void vl::TensorShape::setDimensions(size_t const * newDimensions, size_t newNumDimensions)
{
  assert(newNumDimensions  <= VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS) ;
  for (int k = 0 ; k < newNumDimensions ; ++k) {
    dimensions[k] = newDimensions[k] ;
  }
  numDimensions = newNumDimensions ;
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
: TensorShape(), dataType(VLDT_Float),
  deviceType(VLDT_CPU), memory(NULL), memorySize(0)
{ }

vl::Tensor::Tensor(TensorShape const & shape, DataType dataType,
                   DeviceType deviceType, void * memory, size_t memorySize)
: TensorShape(shape),
dataType(dataType),
deviceType(deviceType),
memory(memory), memorySize(memorySize)
{ }

TensorShape vl::Tensor::getShape() const
{
  return TensorShape(*this) ;
}

vl::DataType vl::Tensor::getDataType() const { return dataType ; }
void * vl::Tensor::getMemory() { return memory ; }
void vl::Tensor::setMemory(void * x) { memory = x ; }
vl::DeviceType vl::Tensor::getDeviceType() const { return deviceType ; }
bool vl::Tensor::isNull() const { return memory == NULL ; }
vl::Tensor::operator bool() const { return !isNull() ; }


