// @file nnsubsample.cu
// @brief Subsampling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnsubsample.hpp"
#include "impl/dispatcher.hpp"
#include "impl/subsample.hpp"
#include "impl/blashelper.hpp"
#include <assert.h>

using namespace std ;
using namespace vl ;
using namespace vl::nn ;

template<vl::DeviceType deviceType, vl::DataType dataType> struct SubsampleForward ;
template<vl::DeviceType deviceType, vl::DataType dataType> struct SubsampleBackward ;

// -------------------------------------------------------------------
//                                                         Forward CPU
// -------------------------------------------------------------------

template<vl::DeviceType deviceType, vl::DataType dataType>
struct SubsampleAndBiasForward
{
  vl::ErrorCode operator()(Subsample &op,
                           Tensor output,
                           Tensor input,
                           Tensor biases)
  {
    assert(output) ;
    assert(input) ;

    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
    type const* allOnesMemory = (type*) op.context.getAllOnes(deviceType, dataType, numOutputPixels) ;

    if (allOnesMemory == NULL) {
      error = op.context.getLastError() ;
      goto done ;
    }

    for (int image = 0 ; image < input.getSize() ; ++image) {
      ptrdiff_t dataOffset = (input.getHeight()*input.getWidth()*input.getDepth()) * image ;
      ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;
      error = vl::impl::subsample<deviceType,type>::forward
      (op.context,
       (type*)output.getMemory() + outputOffset,
       (type const*)input.getMemory() + dataOffset,
       input.getHeight(), input.getWidth(), input.getDepth(),
       op.strideY, op.strideX,
       op.padTop, op.padBottom,
       op.padLeft, op.padRight) ;
      if (error != vl::VLE_Success) { goto done ; }
      if (biases) {
        type alpha = 1 ;
        type beta = 1 ;
        error = vl::impl::blas<deviceType, dataType>::gemm
        (op.context,
         'n', 'n',
         numOutputPixels, biases.getNumElements(), 1,
         alpha,
         allOnesMemory, numOutputPixels,
         (type*)biases.getMemory(), 1,
         beta,
         (type*)output.getMemory() + outputOffset, numOutputPixels) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }
  done:
    return op.context.passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                        Backward CPU
// -------------------------------------------------------------------

template<vl::DeviceType deviceType, vl::DataType dataType>
struct SubsampleAndBiasBackward
{
  vl::ErrorCode operator()(vl::nn::Subsample &op,
                           vl::Tensor derInput,
                           vl::Tensor derBiases,
                           vl::Tensor derOutput)
  {

    assert(derOutput) ;

    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
    type const* allOnesMemory = (type*) op.context.getAllOnes(deviceType, dataType, numOutputPixels) ;

    if (allOnesMemory == NULL) {
      error = op.context.getLastError() ;
      goto done ;
    }

    for (int image = 0 ; image < derInput.getSize() ; ++image) {
      ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

      // Compute derBiases.
      if (derBiases) {
        type alpha = 1 ;
        type beta = (image > 0) ; // Avoids having to clear derOutputs first.
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.context,
         't',
         numOutputPixels, derOutput.getDepth(),
         alpha,
         (type const*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
         allOnesMemory, 1,
         beta,
         (type*)derBiases.getMemory(), 1) ;
        if (error != vl::VLE_Success) { goto done ; }
      }

      // Compute derInput.
      if (derInput) {
        ptrdiff_t derDataOffset = (derInput.getHeight()*derInput.getWidth()*derInput.getDepth()) * image ;
        error = vl::impl::subsample<deviceType,type>::backward
        (op.context,
         (type*)derInput.getMemory() + derDataOffset,
         (type const*)derOutput.getMemory() + derOutputOffset,
         derInput.getHeight(), derInput.getWidth(), derInput.getDepth(),
         op.strideY, op.strideX,
         op.padTop, op.padBottom, op.padLeft, op.padRight) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }
  done:
    return op.context.passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_GPU
//#include "subsample_gpu.cu"
#endif

Subsample::Subsample(vl::Context &context,
                     int strideY, int strideX,
                     int padTop, int padBottom,
                     int padLeft, int padRight)
: context(context),
  strideY(strideY), strideX(strideX),
  padTop(padTop), padBottom(padBottom),
  padLeft(padLeft), padRight(padRight)
{ }

vl::ErrorCode
Subsample::forwardWithBias(vl::Tensor output,
                           vl::Tensor input,
                           vl::Tensor biases)
{
  return dispatch<SubsampleAndBiasForward>()(*this,output,input,biases) ;
}

vl::ErrorCode
Subsample::backwardWithBias(vl::Tensor derInput,
                            vl::Tensor derBiases,
                            vl::Tensor derOutput)
{
  return dispatch<SubsampleAndBiasBackward>()(*this,derInput,derBiases,derOutput) ;
}
