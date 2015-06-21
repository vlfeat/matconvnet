// @file nnbias.cu
// @brief Bias block
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbias.hpp"
#include "impl/nnbias_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnbias_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* Dispatchers                                                     */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnbias_forward(vl::Context& context,
                   vl::Tensor output, double outputMult,
                   vl::Tensor data, double dataMult,
                   vl::Tensor biases, double biasesMult)
{
  vl::Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::nnbias_forward_blas<vl::CPU,float>
      (context, output, outputMult, data, dataMult, biases, biasesMult) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        status = vl::impl::nnbias_forward_cudnn<float>
        (context, output, outputMult, data, dataMult, biases, biasesMult) ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnbias_forward_blas<vl::GPU,float>
      (context, output, outputMult, data, dataMult, biases, biasesMult) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return context.passError(status, "nnbias_forward: ") ;
}

vl::Error
vl::nnbias_backward(vl::Context& context,
                    vl::Tensor derData, double derDataMult,
                    vl::Tensor derBiases, double derBiasesMult,
                    vl::Tensor derOutput, double derOutputMult)
{
  vl::Error status = vlSuccess ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::nnbias_backward_blas<vl::CPU,float>
      (context, derData, derDataMult, derBiases, derBiasesMult, derOutput, derOutputMult) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        status = vl::impl::nnbias_backward_cudnn<float>
        (context, derData, derDataMult, derBiases, derBiasesMult, derOutput, derOutputMult) ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnbias_backward_blas<GPU,float>
      (context, derData, derDataMult, derBiases, derBiasesMult, derOutput, derOutputMult) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return context.passError(status, "nnbias_backward: ") ;
}

