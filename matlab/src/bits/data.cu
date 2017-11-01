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
#include <cmath>

#include <time.h>
#ifdef _MSC_VER
#include "windows.h"
#else
#include <sys/time.h>
#endif

#ifndef NDEBUG
#include <iostream>
#endif

#if ENABLE_GPU
#include "datacu.hpp"
#endif

using namespace vl ;

// -------------------------------------------------------------------
/// MARK: - Helpers
// -------------------------------------------------------------------/

Int vl::gcd(Int a, Int b, Int &u, Int &v)
{
  assert(a >= 0) ;
  assert(b >= 0) ;
  Int u_ = 0 ;
  Int v_ = 1 ;
  u = 1 ;
  v = 0 ;
  while (b > 0) {
    Int tmp ;
    Int q = a / b ;

    tmp = b ;
    b = a - q*b ;
    a = tmp ;

    tmp = u_ ;
    u_ = u - q*u_ ;
    u = tmp ;

    tmp = v_ ;
    v_ = v - q*v_ ;
    v = tmp ;
  }
  return a ;
}

size_t vl::getTime()
{
#ifdef _MSC_VER
  LARGE_INTEGER t ;
  QueryPerformanceFrequency(&t) ;
  return (size_t)(t.QuadPart / 1000ULL) ;
#else
  struct timeval time ;
  int error = gettimeofday(&time, NULL) ;
  assert(error == 0) ;
  return (size_t)time.tv_sec * 1000000 + (size_t)time.tv_usec ;
#endif
}

const char *
vl::getErrorMessage(ErrorCode error)
{
  static char const * messages[] = {
    "Success",
    "Unsupported feature error",
    "CUDA error",
    "CuDNN error",
    "CuBLAS error",
    "Out of memory error",
    "Out of GPU memory error",
    "Illegal argument error",
    "Timeout error",
    "Missing data arror",
    "Illegal message error",
    "Interrupted error",
    "Tensor shape mismatch error",
    "Tensor type mismatch error",
    "Unknown error",
  } ;
  if (error < VLE_Success || error > VLE_Unknown) {
    error = VLE_Unknown ;
  }
  return messages[error] ;
}

static size_t
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

namespace vl { namespace impl {
class Randn
{
public:
  Randn()
  {
    tx[1] = 3.655420419026953 ;
    ty[1] = f(tx[1]) ;
    double A = ty[1] * (tx[1] + 1./tx[1]) ;
    for (int k = 1 ; k < K ; ++k) {
      ty[k] = f(tx[k]) ;
      tx[k+1] = invf(A/tx[k] + ty[k]) ;
    }
    tx[0] = A / ty[1] ;
  }

  double sample() {
    while (true) {
      int k =  rand() % K ;
      double u = (double)rand() / RAND_MAX ;
      double x = tx[k] * (2.*u - 1.) ;
      double absx = fabs(x) ;
      if (absx <= tx[k+1]) { return x ; }
      double v = (double)rand() / RAND_MAX ;
      if (k > 1) {
        double y = v * (ty[k+1] - ty[k]) + ty[k] ;
        if (y <= f(absx)) { return x ; }
      } else {
        double w = (double)rand() / RAND_MAX ;
        double dx = - log(w) / tx[1] ;
        if (- 2.0 * log(v) > dx*dx) {
          double s = (rand() & 0x1) ? +1 : -1 ;
          return s * (tx[1] + dx) ;
        }
      }
    }
  }

private:
  enum { K = 256, } ;
  double tx [K+1] ;
  double ty [K+1] ;

  double f(double x) {
    // 1/sqrt(2*pi)
    return 0.398942280401433 * exp(-0.5 * x * x) ;
  }

  double invf(double y) {
    // sqrt(2*pi)
    return sqrt(-2.0 * log(2.506628274631000 * y));
  }
} ;

} } // namespaces

double vl::randn()
{
  static vl::impl::Randn rnd ;
  return rnd.sample() ;
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

// -------------------------------------------------------------------
/// MARK: - Context
// -------------------------------------------------------------------

vl::Context::Context()
:
lastError(vl::VLE_Success), lastErrorMessage(), cudaHelper(NULL),
logLevel(0)
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

void
vl::Context::setLogLevel(int level)
{
  logLevel = level ;
}

int
vl::Context::getLogLevel() const
{
  return logLevel ;
}

void
vl::Context::log(std::string const &message)
{
  logbook.push_back(message) ;
}

std::string const&
vl::Context::getLastLoggedMessage() const
{
  static std::string const noMessage("No Message") ;
  if (logbook.empty()) {
    return noMessage ;
  }
  return logbook.back() ;
}

std::vector<std::string> const&
vl::Context::getLogbook() const
{
  return logbook ;
}

void
vl::Context::clearLog()
{
  logbook.clear() ;
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
        for (long unsigned i = 0 ; i < size ; ++i) {
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
            <<<divideAndRoundUp((unsigned)size,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS>>>
          ((float*)data, (int)size) ;
        } else {
          setToOnes<double>
            <<<divideAndRoundUp((unsigned)size,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS>>>
          ((double*)data, (int)size) ;
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

// -------------------------------------------------------------------
/// MARK: - TensorShape
// -------------------------------------------------------------------

vl::TensorShape::TensorShape()
: dimensions {}
{ }

vl::TensorShape::TensorShape(TensorShape const & t)
{
  dimensions = t.dimensions ; // copy
}

vl::TensorShape::TensorShape(TensorShape && t)
{
  dimensions = std::move(t.dimensions) ; // move
}

vl::TensorShape& vl::TensorShape::operator= (TensorShape const& ts)
{
  dimensions = ts.dimensions ;
  return *this ;
}

vl::TensorShape::TensorShape(const std::initializer_list<Int> &dims)
{
  assert(dims.size() <= maxNumDimensions) ;
  dimensions = std::vector<Int>(dims) ; // copy
}

vl::TensorShape::TensorShape(const std::vector<Int> &dims)
{
  assert(dims.size() <= maxNumDimensions) ;
  dimensions = dims ; // copy
}

vl::TensorShape::TensorShape(Int height, Int width, Int depth, Int size)
: dimensions {height,width,depth,size}
{ }

void vl::TensorShape::clear()
{
  dimensions.clear() ;
}

void vl::TensorShape::setDimensions(Int const * newDimensions, Int newNumDimensions)
{
  assert(newNumDimensions  <= maxNumDimensions) ;
  dimensions = std::vector<Int>{newDimensions, newDimensions + newNumDimensions} ;
}

void vl::TensorShape::setDimension(Int num, Int dimension)
{
  assert(num >= 0) ;
  assert(num + 1 <= maxNumDimensions) ;
  if (num + 1 > (Int)dimensions.size()) {
    Int x = (getNumElements() > 0) ; // 0 or 1
    for (Int k = (Int)dimensions.size() ; k < num + 1 ; ++k) {
      dimensions.push_back(x) ;
    }
  }
  dimensions[(size_t)num] = dimension ;
}

Int vl::TensorShape::getDimension(Int num) const
{
  assert(num >= 0) ;
  if (num + 1 > (Int)dimensions.size()) {
    return 1 ;
  }
  return dimensions[(size_t)num] ;
}

Int vl::TensorShape::getNumDimensions() const
{
  return (Int)dimensions.size() ;
}

Int const * vl::TensorShape::getDimensionsAsPtr() const
{
  return &dimensions[0] ;
}

std::vector<Int> const& vl::TensorShape::getDimensions() const
{
   return dimensions ;
}

Int vl::TensorShape::getNumElements() const
{
  if (dimensions.size() == 0) {
    return 0 ;
  }
  Int n = 1 ;
  for (size_t k = 0 ; k < dimensions.size() ; ++k) { n *= dimensions[k] ; }
  return n ;
}

Int vl::TensorShape::getHeight() const { return getDimension(0) ; }
Int vl::TensorShape::getWidth() const { return getDimension(1) ; }
Int vl::TensorShape::getNumChannels() const { return getDimension(2) ; }
Int vl::TensorShape::getCardinality() const { return getDimension(3) ; }

void vl::TensorShape::setHeight(Int x) { setDimension(0,x) ; }
void vl::TensorShape::setWidth(Int x) { setDimension(1,x) ; }
void vl::TensorShape::setDepth(Int x) { setDimension(2,x) ; }
void vl::TensorShape::setSize(Int x) { setDimension(3,x) ; }
bool vl::TensorShape::isEmpty() const { return getNumElements() == 0 ; }

bool vl::operator== (vl::TensorShape const & a, vl::TensorShape const & b)
{
  Int n = a.getNumDimensions() ;
  if (b.getNumDimensions() != n) { return false ; }
  for (unsigned k = 0 ; k < a.getNumDimensions() ; ++k) {
    if (a.getDimension(k) != b.getDimension(k)) { return false ; }
  }
  return true ;
}

void vl::TensorShape::reshape(Int newNumDimensions)
{
  assert(newNumDimensions <= maxNumDimensions) ;
  Int n = getNumElements() ;
  if (newNumDimensions > 0) {
    setDimension(newNumDimensions - 1, 1) ;
    dimensions.resize((size_t)newNumDimensions) ;
    Int m = getNumElements() ;
    if (m) {
      dimensions[(size_t)newNumDimensions - 1] *= (n / m) ;
    } else if (n == 0) {
      dimensions[(size_t)newNumDimensions - 1] = 0  ;
    }
  } else {
    dimensions.resize(0) ;
  }
}

void vl::TensorShape::reshape(TensorShape const & newShape)
{
  operator=(newShape) ;
}

// -------------------------------------------------------------------
/// MARK: - Tensor
// -------------------------------------------------------------------

vl::Tensor::Tensor()
: TensorShape(), dataType(VLDT_Float),
  deviceType(VLDT_CPU), memory(NULL), memorySize(0)
{ }

vl::Tensor::Tensor(Tensor &&t)
: TensorShape(t),
dataType(t.dataType),
deviceType(t.deviceType), memory(t.memory), memorySize(t.memorySize)
{
  t.memory = NULL ;
  t.memorySize = 0 ;
}

vl::Tensor::Tensor(vl::Tensor const &t)
: TensorShape(t), dataType(t.dataType),
deviceType(t.deviceType), memory(t.memory), memorySize(t.memorySize)
{ }

vl::Tensor& vl::Tensor::operator= (vl::Tensor::Tensor const &t)
{
  TensorShape::operator=(t) ;
  dataType = t.dataType ;
  deviceType = t.deviceType ;
  memory = t.memory ;
  memorySize = t.memorySize ;
  return *this ;
}

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
void const * vl::Tensor::getMemory() const { return memory ; }
void vl::Tensor::setMemory(void * x) { memory = x ; }
vl::DeviceType vl::Tensor::getDeviceType() const { return deviceType ; }
bool vl::Tensor::isNull() const { return memory == NULL ; }
vl::Tensor::operator bool() const { return !isNull() ; }


