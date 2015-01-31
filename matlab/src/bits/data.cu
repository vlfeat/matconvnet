//
//  data.cu
//  matconv
//
//  Created by Andrea Vedaldi on 31/01/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "data.hpp"
#include <assert.h>
#include <iostream>

#if ENABLE_GPU
#include "datacu.hpp"
#endif

using namespace vl ;

/* -------------------------------------------------------------------
 * Context
 * ---------------------------------------------------------------- */

vl::Context::Context()
: cpuWorkspace(0), cpuWorkspaceSize(0),
gpuWorkspace(0), gpuWorkspaceSize(0),
cudaHelper(0)
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
  std::cout<<"Context::reset()"<<std::endl ;
}

vl::Context::~Context()
{
  reset() ;
  std::cout<<"Context::~Context()"<<std::endl ;
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
        cpuWorkspaceSize = size ;
        //std::cout<<"Context: allocated "<<size<< " bytes of CPU memory."<<std::endl ;
      }
      //std::cout<<"Context: requested "<<size<< " bytes of CPU memory."<<std::endl ;
      return cpuWorkspace ;

    case GPU:
#if ENABLE_GPU
      if (gpuWorkspaceSize < size) {
        clearWorkspace(GPU) ;
        cudaMalloc(&gpuWorkspace, size) ;
        gpuWorkspaceSize = size ;
        //std::cout<<"Context: allocated "<<size<< " bytes of GPU memory."<<std::endl ;
      }
      //std::cout<<"Context: requested "<<size<< " bytes of GPU memory."<<std::endl ;
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
template<typename type>
__global__ void setToOnes (type * data, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < size) data[index] = type(1.0) ;
}
#endif

void *
vl::Context::getAllOnes(Device device, Type type, size_t size)
{
  int typeSize = (type == FLOAT) ? sizeof(float) : sizeof(double) ;

  switch (device) {
    case CPU:
      if (cpuAllOnesSize < size || cpuAllOnesType != type) {
        clearAllOnes(CPU) ;
        cpuAllOnes = malloc(size * typeSize) ;
        cpuAllOnesType = type ;
        cpuAllOnesSize = size ;
        for (int i = 0 ; i < size ; ++i) {
          if (type == FLOAT) {
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
        clearAllOnes(GPU) ;
        cudaMalloc(&gpuAllOnes, size * typeSize) ;
        gpuAllOnesType = type ;
        gpuAllOnesSize = size ;
        if (type == FLOAT) {
          setToOnes<float>
          <<<divideUpwards(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
          ((float*)gpuAllOnes, size) ;
        } else {
          setToOnes<double>
          <<<divideUpwards(size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
          ((double*)gpuAllOnes, size) ;
        }
      }
      return gpuAllOnes ;
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
bool vl::Tensor::isNull() const { return memory != NULL ; }
vl::Tensor::operator bool() const { return isNull() ; }

