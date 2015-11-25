// @file nnsubsample.cu
// @brief Subsampling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnsubsample.hpp"
#include "impl/subsample.hpp"
#include "impl/blashelper.hpp"
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* Implementations                                                  */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type> vl::Error
nnsubsample_forward_impl(Context& context,
                         Tensor output,
                         Tensor data,
                         Tensor biases,
                         int strideY, int strideX,
                         int padTop, int padBottom,
                         int padLeft, int padRight)
{
  assert(output) ;
  assert(data) ;

  vl::Error error ;

  ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
  type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                         get_vl_type<type>(),
                                                         numOutputPixels) ;
  if (allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < data.getSize() ; ++image) {
    ptrdiff_t dataOffset = (data.getHeight()*data.getWidth()*data.getDepth()) * image ;
    ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;
    error = vl::impl::subsample_forward<arch,type>(context,
                                                   output.getMemory() + outputOffset,
                                                   data.getMemory() + dataOffset,
                                                   data.getHeight(), data.getWidth(), data.getDepth(),
                                                   strideY, strideX,
                                                   padTop, padBottom, padLeft, padRight) ;
    if (error != vl::vlSuccess) { goto done ; }
    if (biases) {
      type alpha = 1 ;
      type beta = 1 ;
      error = gemm<arch,type>(context,
                              'n', 'n',
                              numOutputPixels, biases.getNumElements(), 1,
                              alpha,
                              allOnesMemory, numOutputPixels,
                              (type*)biases.getMemory(), 1,
                              beta,
                              (type*)output.getMemory() + outputOffset, numOutputPixels) ;
      if (error != vl::vlSuccess) { goto done ; }
    }
  }
done:
  return context.passError(error, "nnsubsample_forward_impl<>: ") ;
}

template<vl::Device arch, typename type> vl::Error
nnsubsample_backward_impl(Context& context,
                          Tensor derData,
                          Tensor derBiases,
                          Tensor derOutput,
                          int strideY, int strideX,
                          int padTop, int padBottom,
                          int padLeft, int padRight)
{
  assert(derOutput) ;

  vl::Error error ;

  ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
  type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                         get_vl_type<type>(),
                                                         numOutputPixels) ;
  if (allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < derData.getSize() ; ++image) {
    ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

    /* compute derBiases = dz/dbias */
    if (derBiases) {
      type alpha = 1 ;
      type beta = (image > 0) ; /* this saves init. the output array with 0 */
      error = gemv<arch,type>(context,
                      't',
                      numOutputPixels, derOutput.getDepth(),
                      alpha,
                      derOutput.getMemory() + derOutputOffset, numOutputPixels,
                      allOnesMemory, 1,
                      beta,
                      derBiases.getMemory(), 1) ;
      if (error != vl::vlSuccess) { goto done ; }
    }

    /* compute derData = dz/dx */
    if (derData) {
      ptrdiff_t derDataOffset = (derData.getHeight()*derData.getWidth()*derData.getDepth()) * image ;
      error = vl::impl::subsample_backward<arch,type>(context,
                                              derData.getMemory() + derDataOffset,
                                              derOutput.getMemory() + derOutputOffset,
                                              derData.getHeight(), derData.getWidth(), derData.getDepth(),
                                              strideY, strideX,
                                              padTop, padBottom, padLeft, padRight) ;
      if (error != vl::vlSuccess) { goto done ; }
    }
  }
done:
  return context.passError(error, "nnsubsample_forward_impl<>: ") ;
}

/* ---------------------------------------------------------------- */
/* Dispatchers                                                     */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnsubsample_forward(Context& context,
                        Tensor output,
                        Tensor data,
                        Tensor biases,
                        int strideY, int strideX,
                        int padTop, int padBottom,
                        int padLeft, int padRight)
{
  vl::Error error = vl::vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      error = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      error = nnsubsample_forward_impl<CPU,float>
      (context, output, data, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
      error = nnsubsample_forward_impl<GPU,float>
      (context, output, data, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;
#endif
  }
  return context.passError(error, "nnsubsample_forward: ") ;
}

vl::Error
vl::nnsubsample_backward(vl::Context& context,
                         vl::Tensor derData,
                         vl::Tensor derBiases,
                         vl::Tensor derOutput,
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
      status = nnsubsample_backward_impl<CPU,float>
      (context,
       derData, derBiases, derOutput,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
      status = nnsubsample_backward_impl<GPU,float>
      (context,
       derData, derBiases, derOutput,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;
#endif
  }
  return status ;
}
