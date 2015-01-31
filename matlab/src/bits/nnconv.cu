//
//  nnconv.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 04/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

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

int vl::nnconv_forward(Context& context,
                       Tensor output,
                       Tensor data,
                       Tensor filters,
                       Tensor biases,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight)
{
  int status = 0 ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::ERROR ;
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
      if (context.getCudaHelper().isCudnnActive()) {
        status = vl::impl::nnconv_forward_cudnn<float>
        (context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
        if (status == vl::SUCCESS) { return status ; }
        if (status != vl::UNSUPPORTED) { return status ; }
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
  return status ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */

int vl::nnconv_backward(Context& context,
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
  int status = 0 ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::ERROR ;
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
      if (context.getCudaHelper().isCudnnActive()) {
        status = vl::impl::nnconv_backward_cudnn<float>
        (context,
         derData, derFilters, derBiases,
         data, filters, derOutput,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
        if (status == vl::SUCCESS || status != vl::UNSUPPORTED) break ;
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
  return status ;
}
