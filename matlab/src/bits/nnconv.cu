// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014-15 Andrea Vedaldi and Max Jaderberg
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnconv.hpp"
#include "impl/nnconv_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnconv_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnconv_forward(Context& context,
                   Tensor output,
                   Tensor data,
                   Tensor filters,
                   Tensor biases,
                   int strideY, int strideX,
                   int padTop, int padBottom,
                   int padLeft, int padRight)
{
  vl::Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::nnconv_forward_blas<CPU,float>
      (context, output, data, filters, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        status = vl::impl::nnconv_forward_cudnn<float>
        (context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnconv_forward_blas<GPU,float>
      (context, output, data, filters, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return status ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnconv_backward(Context& context,
                    Tensor derData,
                    Tensor derFilters,
                    Tensor derBiases,
                    Tensor data,
                    Tensor filters,
                    Tensor derOutput,
                    int strideY, int strideX,
                    int padTop, int padBottom,
                    int padLeft, int padRight)
{
  vl::Error status = vl::vlSuccess ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::nnconv_backward_blas<CPU,float>
      (context,
       derData, derFilters, derBiases,
       data, filters, derOutput,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        status = vl::impl::nnconv_backward_cudnn<float>
        (context,
         derData, derFilters, derBiases,
         data, filters, derOutput,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnconv_backward_blas<GPU,float>
      (context,
       derData, derFilters, derBiases,
       data, filters, derOutput,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return status ;
}
