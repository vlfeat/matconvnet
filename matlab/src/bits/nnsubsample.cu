//
//  nnsubsample.cu
//  matconv
//
//  Created by Andrea Vedaldi on 07/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "nnsubsample.hpp"
#include "impl/subsample.hpp"
#include "impl/blashelper.hpp"
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* Implementations                                                  */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type>
int nnsubsample_forward_impl(Context& context,
                             Tensor output,
                             Tensor data,
                             Tensor biases,
                             int strideY, int strideX,
                             int padTop, int padBottom,
                             int padLeft, int padRight)
{
  assert(output) ;
  assert(data) ;

  ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
  type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                         get_type_id<type>(),
                                                         numOutputPixels) ;

  for (int image = 0 ; image < data.getSize() ; ++image) {
    ptrdiff_t dataOffset = (data.getHeight()*data.getWidth()*data.getDepth()) * image ;
    ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;
    vl::impl::subsample_forward<arch,type>(context,
                                           output.getMemory() + outputOffset,
                                           data.getMemory() + dataOffset,
                                           data.getHeight(), data.getWidth(), data.getDepth(),
                                           strideY, strideX,
                                           padTop, padBottom, padLeft, padRight) ;
    if (biases) {
      type alpha = 1 ;
      type beta = 1 ;
      gemm<arch,type>(context,
                      'n', 'n',
                      numOutputPixels, biases.getNumElements(), 1,
                      alpha,
                      allOnesMemory, numOutputPixels,
                      (type*)biases.getMemory(), 1,
                      beta,
                      (type*)output.getMemory() + outputOffset, numOutputPixels) ;
    }
  }

  return 0 ;
}

template<vl::Device arch, typename type>
int nnsubsample_backward_impl(Context& context,
                              Tensor derData,
                              Tensor derBiases,
                              Tensor derOutput,
                              int strideY, int strideX,
                              int padTop, int padBottom,
                              int padLeft, int padRight)
{
  assert(derOutput) ;

  ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
  type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                         get_type_id<type>(),
                                                         numOutputPixels) ;


  for (int image = 0 ; image < derData.getSize() ; ++image) {
    ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

    /* compute derBiases = dz/dbias */
    if (derBiases) {
      type alpha = 1 ;
      type beta = (image > 0) ; /* this saves init. the output array with 0 */
      gemv<arch,type>(context,
                      't',
                      numOutputPixels, derOutput.getDepth(),
                      alpha,
                      derOutput.getMemory() + derOutputOffset, numOutputPixels,
                      allOnesMemory, 1,
                      beta,
                      derBiases.getMemory(), 1) ;
    }

    /* compute derData = dz/dx */
    if (derData) {
      ptrdiff_t derDataOffset = (derData.getHeight()*derData.getWidth()*derData.getDepth()) * image ;
      vl::impl::subsample_backward<arch,type>(context,
                                              derData.getMemory() + derDataOffset,
                                              derOutput.getMemory() + derOutputOffset,
                                              derData.getHeight(), derData.getWidth(), derData.getDepth(),
                                              strideY, strideX,
                                              padTop, padBottom, padLeft, padRight) ;
    }
  }
  return 0 ;
}

/* ---------------------------------------------------------------- */
/* Dispatchers                                                     */
/* ---------------------------------------------------------------- */

int vl::nnsubsample_forward(Context& context,
                            Tensor output,
                            Tensor data,
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
      status = nnsubsample_forward_impl<CPU,float>
      (context, output, data, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
      status = nnsubsample_forward_impl<GPU,float>
      (context, output, data, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;
#endif
  }
  return status ;
}

int vl::nnsubsample_backward(vl::Context& context,
                             vl::Tensor derData,
                             vl::Tensor derBiases,
                             vl::Tensor derOutput,
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
