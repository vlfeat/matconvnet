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

namespace vl { namespace impl {

  template <typename type>
  struct operations<vl::GPU, type>
  {
    typedef type data_type ;

    static vl::Error
    copy(data_type * dest,
         data_type const * src,
         size_t numElements)
    {
      cudaMemcpy(dest, src, numElements * sizeof(data_type), cudaMemcpyDeviceToDevice) ;
      return vlSuccess ;
    }

    static vl::Error
    fill(data_type * dest,
         size_t numElements,
         data_type value)
    {
      fill_kernel <data_type>
      <<<divideUpwards(numElements, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
      (dest, numElements, value) ;

      cudaError_t error = cudaGetLastError() ;
      if (error != cudaSuccess) {
        return vlErrorCuda ;
      }
      return vlSuccess ;
    }
  } ;

} }

template struct vl::impl::operations<vl::GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::operations<vl::GPU, double> ;
#endif