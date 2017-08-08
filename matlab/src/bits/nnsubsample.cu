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
#include "impl/blashelper.hpp"
#include <cassert>
#include <cstring>

using namespace std ;
using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<vl::DeviceType deviceType, vl::DataType dataType> struct SubsampleForward ;
template<vl::DeviceType deviceType, vl::DataType dataType> struct SubsampleBackward ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct SubsampleForward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(Subsample &op,
                           Tensor &output,
                           Tensor const &input)
  {
    assert(output) ;
    assert(input) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = input.getWidth() ;
    auto height = input.getHeight() ;
    auto depth = input.getDepth() ;
    auto size = input.getSize() ;
    auto inputData = (type*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;
    auto outputWidth = (width + (op.padLeft + op.padRight) - 1)/op.strideX + 1 ;
    auto outputHeight = (height + (op.padTop + op.padBottom) - 1)/op.strideY + 1 ;

    assert(outputWidth == output.getWidth()) ;
    assert(outputHeight == output.getHeight()) ;

    for (int z = 0; z < depth * size ; ++z) {
      for (int x = 0; x < outputWidth; ++x) {
        for (int y = 0; y < outputHeight; ++y) {
          int x1 = x * (int)op.strideX - (int)op.padLeft ;
          int y1 = y * (int)op.strideY - (int)op.padTop ;
          type value = 0 ;
          if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            value = inputData[x1 * height + y1] ;
          }
          outputData[x * outputHeight + y] = value ;
        }
      }
      inputData += width*height ;
      outputData += outputWidth*outputHeight ;
    }
    return VLE_Success ;
  }
} ;

template<vl::DeviceType deviceType, vl::DataType dataType>
struct SubsampleAndBiasForward
{
  vl::ErrorCode operator()(Subsample &op,
                           Tensor &output,
                           Tensor const &input,
                           Tensor const &biases)
  {
    assert(output) ;
    assert(input) ;

    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    error = SubsampleForward<deviceType,dataType>()(op,output,input) ;
    if (error != VLE_Success) { return error ; }

    ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
    type const* allOnesMemory = (type*) op.context.getAllOnes(deviceType, dataType, numOutputPixels) ;

    if (allOnesMemory == NULL) {
      error = op.context.getLastError() ;
      goto done ;
    }

    for (int image = 0 ; image < input.getSize() ; ++image) {
      ptrdiff_t dataOffset = (input.getHeight()*input.getWidth()*input.getDepth()) * image ;
      ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;
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
//                                                            Backward
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct SubsampleBackward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(Subsample &op,
                           Tensor &derInput,
                           Tensor const &derOutput)
  {
    assert(derInput) ;
    assert(derOutput) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = derInput.getWidth() ;
    auto height = derInput.getHeight() ;
    auto depth = derInput.getDepth() ;
    auto size = derInput.getSize() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derOutputData = (type*)derOutput.getMemory() ;
    auto outputWidth = (width + (op.padLeft + op.padRight) - 1)/op.strideX + 1 ;
    auto outputHeight = (height + (op.padTop + op.padBottom) - 1)/op.strideY + 1 ;

    assert(outputWidth == derOutput.getWidth()) ;
    assert(outputHeight == derOutput.getHeight()) ;

    memset(derInputData, 0, sizeof(type) * width * height * depth * size) ;

    for (int z = 0; z < depth * size; ++z) {
      for (int px = 0; px < outputWidth; ++px) {
        for (int py = 0; py < outputHeight; ++py) {
          int x1 = px * (int)op.strideX - (int)op.padLeft ;
          int y1 = py * (int)op.strideY - (int)op.padTop ;
          if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            derInputData[x1 * height + y1] = derOutputData[px * outputHeight + py] ;
          }
        }
      }
      derInputData += width*height ;
      derOutputData += outputWidth*outputHeight ;
    }
    return VLE_Success ;
  }
} ;

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

    // Compute derInput.
    if (derInput) {
      error = SubsampleBackward<deviceType,dataType>()(op,derInput,derOutput) ;
      if (error != VLE_Success) { return error ; }
    }

    // Compute derBiases.
    if (derBiases) {
      ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
      type const* allOnesMemory = (type*) op.context.getAllOnes(deviceType, dataType, numOutputPixels) ;

      if (allOnesMemory == NULL) {
        error = op.context.getLastError() ;
        goto done ;
      }

      for (int image = 0 ; image < derInput.getSize() ; ++image) {
        ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;
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
    }

  done:
    return op.context.passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_GPU
#include "nnsubsample_gpu.cu"
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
Subsample::forwardWithBias(vl::Tensor &output,
                           vl::Tensor const &input,
                           vl::Tensor const &biases)
{
  return dispatch<SubsampleAndBiasForward>()(*this,output,input,biases) ;
}

vl::ErrorCode
Subsample::backwardWithBias(vl::Tensor &derInput,
                            vl::Tensor &derBiases,
                            vl::Tensor const &derOutput)
{
  return dispatch<SubsampleAndBiasBackward>()(*this,derInput,derBiases,derOutput) ;
}
