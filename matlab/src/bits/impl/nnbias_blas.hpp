// @file nnbias_blas.hpp
// @brief Bias block BLAS implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbias_blas__
#define __vl__nnbias_blas__

#include <assert.h>
#include "blashelper.hpp"

namespace vl { namespace impl {

  template<vl::Device arch, typename type> inline vl::Error
  nnbias_forward_blas(vl::Context& context,
                      vl::Tensor output, double outputMult,
                      vl::Tensor data, double dataMult,
                      vl::Tensor biases, double biasesMult) ;

  template<vl::Device arch, typename type> inline vl::Error
  nnbias_backward_blas(vl::Context& context,
                       vl::Tensor derData, double derDataMult,
                       vl::Tensor derBiases, double derBiasesMult,
                       vl::Tensor derOutput, double derOutputMult) ;

} }


template<vl::Device arch, typename type> inline vl::Error
vl::impl::nnbias_forward_blas(vl::Context& context,
                              vl::Tensor output, double outputMult,
                              vl::Tensor data, double dataMult,
                              vl::Tensor biases, double biasesMult)
{
  vl::Error error ;
  ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
  type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                         get_vl_type<type>(),
                                                         numOutputPixels) ;
  if (allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < output.getSize() ; ++image) {
    ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;
    double alpha = outputMult ;

    if (biases) {
      error = gemm<arch,type>(context,
                              'n', 'n',
                              numOutputPixels, biases.getNumElements(), 1,
                              biasesMult,
                              allOnesMemory, numOutputPixels,
                              (type*)biases.getMemory(), 1,
                              alpha,
                              (type*)output.getMemory() + outputOffset, numOutputPixels) ;
      if (error != vl::vlSuccess) { goto done ; }
      alpha = 1 ;
    }

    if (data) {
      assert(false) ; // not implemented
      if (error != vl::vlSuccess) { goto done ; }
    }
  }
done:
  return context.passError(error, "nnbias_forward_blas<>: ") ;
}

template<vl::Device arch, typename type> inline vl::Error
vl::impl::nnbias_backward_blas(vl::Context& context,
                               vl::Tensor derData, double derDataMult,
                               vl::Tensor derBiases, double derBiasesMult,
                               vl::Tensor derOutput, double derOutputMult)
{
  vl::Error error ;
  type const* allOnesMemory = NULL ;

  // for all derivatives
  assert(derOutput) ;
  ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;

  if (derBiases) {
    // for derivative w.r.t. bias
    allOnesMemory = (type*) context.getAllOnes(arch,
                                               get_vl_type<type>(),
                                               numOutputPixels) ;
    if (allOnesMemory == NULL) {
      error = context.getLastError() ;
      goto done ;
    }
  }

  if (derData) {
    // for derivative w.r.t. data
    assert(false) ; // not implemented
  }

  for (int image = 0 ; image < derOutput.getSize() ; ++image) {

    ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

    /* compute derData dz/dbias */
    if (derBiases) {
      // has derBiases, derOutput
      error = gemv<arch,type>(context,
                              't',
                              numOutputPixels, derOutput.getDepth(),
                              derOutputMult, /* alpha */
                              derOutput.getMemory() + derOutputOffset, numOutputPixels,
                              allOnesMemory, 1,
                              (image == 0) ? derBiasesMult : 1.0, /* beta */
                              derBiases.getMemory(), 1) ;
      if (error != vl::vlSuccess) { return error ; }
    }

    /* compute derData dz/dx */
    if (derData) {
      // not implemented
      if (error != vl::vlSuccess) { return error ; }
    }
  }

done:
  return context.passError(error, "nnbias_backward_blas<>: ") ;
}

#endif /* defined(__vl__nnbias_blas__) */
