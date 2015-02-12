//
//  nnpooling.cu
//
//
//  Created by Andrea Vedaldi on 09/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "nnpooling.hpp"
#include "impl/pooling.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
//#include "impl/pooling_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnpooling_forward */
/* ---------------------------------------------------------------- */

Error
vl::nnpooling_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      PoolingMethod method,
                      int poolHeight, int poolWidth,
                      int strideY, int strideX,
                      int padTop, int padBottom,
                      int padLeft, int padRight)
{
  Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      switch (method) {
        default:
          assert(false) ;
          return vl::vlErrorUnknown ;
        case vl::AVERAGE:
          status = vl::impl::pooling_average_forward<CPU,float>
          ((float*)output.getMemory(), (float const*)data.getMemory(),
           data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break ;
        case vl::MAX:
          status = vl::impl::pooling_max_forward<CPU,float>
          ((float*)output.getMemory(), (float const*)data.getMemory(),
           data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break;
      }
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().isCudnnEnabled()) {
        /*
         status = vl::impl::nnpooling_forward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (status == vl::vlSuccess) { return status ; }
         if (status != vl::UNSUPPORTED) { return status ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      switch (method) {
        default:
          assert(false) ;
          return vl::vlErrorUnknown ;
        case vl::AVERAGE:
          status = vl::impl::pooling_average_forward<GPU,float>
          ((float*)output.getMemory(), (float const*)data.getMemory(),
           data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break ;
        case vl::MAX:
          status = vl::impl::pooling_max_forward<GPU,float>
          ((float*)output.getMemory(), (float const*)data.getMemory(),
           data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break;
      }
      if (status == vlErrorCuda) {
        context.getCudaHelper().catchCudaError("pooling_*_forward") ;
      }
      break ;
#endif
  }
  return context.setError(status, "nnpooling_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                               nnpooling_backward */
/* ---------------------------------------------------------------- */

Error
vl::nnpooling_backward(Context& context,
                       Tensor derData,
                       Tensor data,
                       Tensor derPooled,
                       PoolingMethod method,
                       int poolHeight, int poolWidth,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight)
{
  vl::Error status = vlSuccess ;
  switch (derData.getMemoryType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      switch (method) {
        default:
          assert(false) ;
          return vl::vlErrorUnknown ;
        case vl::AVERAGE:
          status = vl::impl::pooling_average_backward<CPU,float>
          ((float*)derData.getMemory(), (float const*)derPooled.getMemory(),
           derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break ;
        case vl::MAX:
          status = vl::impl::pooling_max_backward<CPU,float>
          ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derPooled.getMemory(),
           derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break ;
      }
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().isCudnnEnabled()) {
        /*
         status = vl::impl::nnpooling_backward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (status == vl::vlSuccess) { return status ; }
         if (status != vl::UNSUPPORTED) { return status ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      switch (method) {
        default:
          assert(false) ;
          return vl::vlErrorUnknown ;
        case vl::AVERAGE:
          status = vl::impl::pooling_average_backward<GPU,float>
          ((float*)derData.getMemory(), (float const*)derPooled.getMemory(),
           derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break ;
        case vl::MAX:
          status = vl::impl::pooling_max_backward<GPU,float>
          ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derPooled.getMemory(),
           derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
           poolHeight, poolWidth,
           strideY, strideX,
           padTop, padBottom,
           padLeft, padRight) ;
          break ;
      }
      if (status == vlErrorCuda) {
        context.getCudaHelper().catchCudaError("pooling_*_backward") ;
      }
      break ;
#endif
  }
  return context.setError(status, "nnpooling_backward") ;
}
