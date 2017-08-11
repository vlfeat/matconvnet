// @file copy_gpu.cu
// @brief Copy and other data operations (GPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "copy.hpp"
#include "../datacu.hpp"
#include <string.h>

template<typename type> __global__ void
fill_kernel (type * data, type value, size_t size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < size) data[index] = value ;
}

template<typename type> __global__ void
copy_kernel (type *dst, type const *src, size_t size, type mult)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < size) dst[index] = mult * src[index] ;
}

namespace vl { namespace impl {

  template <typename type>
  struct operations<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    copy(type * dst,
         type const * src,
         size_t numElements,
         double mult)
    {
      if (mult == 1.0) {
        cudaMemcpy(dst, src, numElements * sizeof(type), cudaMemcpyDeviceToDevice) ;
      } else {
        copy_kernel <type>
        <<<divideAndRoundUp(numElements, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
        (dst, src, numElements, mult) ;
        cudaError_t error = cudaGetLastError() ;
        if (error != cudaSuccess) {
          return VLE_Cuda ;
        }
      }
      return VLE_Success ;
    }

    static vl::ErrorCode
    fill(type * dst,
         size_t numElements,
         type value)
    {
      fill_kernel <type>
      <<<divideAndRoundUp(numElements, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
      (dst, numElements, value) ;

      cudaError_t error = cudaGetLastError() ;
      if (error != cudaSuccess) {
        return VLE_Cuda ;
      }
      return VLE_Success ;
    }
  } ;

} }

template struct vl::impl::operations<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::operations<vl::VLDT_GPU, double> ;
#endif
