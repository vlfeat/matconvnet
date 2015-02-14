// @file nnnormalize.cu
// @brief Normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnnormalize.hpp"
#include "impl/normalize.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
//#include "impl/normalize_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                              nnnormalize_forward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnnormalize_forward(vl::Context& context,
                        vl::Tensor output,
                        vl::Tensor data,
                        size_t normDetph,
                        double kappa, double alpha, double beta)
{
  vl::Error status = vl::vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      status = vl::impl::normalize_forward<vl::CPU,float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       normDetph, kappa, alpha, beta) ;
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         status = vl::impl::nnnormalize_forward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (status == vl::vlSuccess) { return status ; }
         if (status != vl::UNSUPPORTED) { return status ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::normalize_forward<vl::GPU,float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       normDetph, kappa, alpha, beta) ;
      if (status != vl::vlSuccess) { context.getCudaHelper().catchCudaError("normalize_forward<GPU,float>") ; }
      break ;
#endif
  }
  if (status != vl::vlSuccess) {
    context.setError(status, "normalize_forward") ;
  }
  return status ;
}

/* ---------------------------------------------------------------- */
/*                                             nnnormalize_backward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnnormalize_backward(vl::Context& context,
                         vl::Tensor derData,
                         vl::Tensor data,
                         vl::Tensor derOutput,
                         size_t normDetph,
                         double kappa, double alpha, double beta)
{
  vl::Error status = vl::vlSuccess ;
  switch (derData.getMemoryType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      status = vl::impl::normalize_backward<vl::CPU,float>
      ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derOutput.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       normDetph, kappa, alpha, beta) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         status = vl::impl::nnnormalize_backward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (status == vl::vlSuccess) { return status ; }
         if (status != vl::UNSUPPORTED) { return status ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::normalize_backward<vl::GPU,float>
      ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derOutput.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       normDetph, kappa, alpha, beta) ;
      if (status != vl::vlSuccess) { context.getCudaHelper().catchCudaError("normalize_backward<GPU,float>") ; }
      break ;
#endif
  }
  if (status != vl::vlSuccess) {
    context.setError(status, "normalize_backward") ;
  }
  return status ;
}
