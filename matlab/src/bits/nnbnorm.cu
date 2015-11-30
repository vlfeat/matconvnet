// @file nnbnorm.cu
// @brief Batch normalization block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbnorm.hpp"
#include "impl/bnorm.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnbnorm_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor moments,
                    vl::Tensor data,
                    vl::Tensor multipliers,
                    vl::Tensor biases,
                    float epsilon)
{
  vl::Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::bnorm<vl::CPU,float>::forward
      (context,
       (float*)output.getMemory(),
       (float*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float*)multipliers.getMemory(),
       (float*)biases.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       epsilon);
      break ;

#if ENABLE_GPU
    case vl::GPU:
      status = vl::impl::bnorm<vl::GPU,float>::forward
      (context,
       (float*)output.getMemory(),
       (float*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float*)multipliers.getMemory(),
       (float*)biases.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       epsilon);
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("nnbnorm_*_forward")) ;
      }
      break;
#endif
  }
  return context.passError(status, "nnbnorm_forward") ;
}

vl::Error
vl::nnbnorm_forward_given_moments(vl::Context& context,
                                  vl::Tensor output,
                                  vl::Tensor moments,
                                  vl::Tensor data,
                                  vl::Tensor multipliers,
                                  vl::Tensor biases)
{
  vl::Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::bnorm<vl::CPU,float>::forward_given_moments
      (context,
       (float*)output.getMemory(),
       (float const*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float*)multipliers.getMemory(),
       (float*)biases.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize()) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      status = vl::impl::bnorm<vl::GPU,float>::forward_given_moments
      (context,
       (float*)output.getMemory(),
       (float const*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float*)multipliers.getMemory(),
       (float*)biases.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize());
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("nnbnorm_*_forward")) ;
      }
      break;
#endif
  }
  return context.passError(status, "nnbnorm_forward_given_moments") ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnbnorm_backward(Context& context,
                     vl::Tensor derData,
                     vl::Tensor derMultipliers,
                     vl::Tensor derBiases,
                     vl::Tensor moments,
                     vl::Tensor data,
                     vl::Tensor multipliers,
                     vl::Tensor biases,
                     vl::Tensor derOutput,
                     float epsilon)
{
  vl::Error status = vl::vlSuccess ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::bnorm<vl::CPU,float>::backward
      (context,
       (float*)derData.getMemory(),
       (float*)derMultipliers.getMemory(),
       (float*)derBiases.getMemory(),
       (float*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float const*)multipliers.getMemory(),
       (float const*)biases.getMemory(),
       (float const*)derOutput.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       epsilon);
      break ;

#if ENABLE_GPU
    case vl::GPU:
      status = vl::impl::bnorm<vl::GPU,float>::backward
      (context,
       (float*)derData.getMemory(),
       (float*)derMultipliers.getMemory(),
       (float*)derBiases.getMemory(),
       (float*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float const*)multipliers.getMemory(),
       (float const*)biases.getMemory(),
       (float const*)derOutput.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       epsilon);

      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("bnorm_*_backward")) ;
      }
      break;
#endif
  }
  return context.passError(status, "nnbnorm_backward") ;
}

vl::Error
vl::nnbnorm_backward_given_moments(Context& context,
                                   vl::Tensor derData,
                                   vl::Tensor derMultipliers,
                                   vl::Tensor derBiases,
                                   vl::Tensor moments,
                                   vl::Tensor data,
                                   vl::Tensor multipliers,
                                   vl::Tensor biases,
                                   vl::Tensor derOutput,
                                   float epsilon)
{
  vl::Error status = vl::vlSuccess ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::bnorm<vl::CPU,float>::backward_given_moments
      (context,
       (float*)derData.getMemory(),
       (float*)derMultipliers.getMemory(),
       (float*)derBiases.getMemory(),
       (float*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float const*)multipliers.getMemory(),
       (float const*)biases.getMemory(),
       (float const*)derOutput.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       epsilon);
      break ;

#if ENABLE_GPU
    case vl::GPU:
      status = vl::impl::bnorm<vl::GPU,float>::backward_given_moments
      (context,
       (float*)derData.getMemory(),
       (float*)derMultipliers.getMemory(),
       (float*)derBiases.getMemory(),
       (float const*)moments.getMemory(),
       (float const*)data.getMemory(),
       (float const*)multipliers.getMemory(),
       (float const*)biases.getMemory(),
       (float const*)derOutput.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(),
       epsilon);

      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("bnorm_*_backward")) ;
      }
      break;
#endif
  }
  return context.passError(status, "nnbnorm_backward_given_moments") ;
}
