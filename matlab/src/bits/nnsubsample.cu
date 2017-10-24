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
  vl::ErrorCode operator()(Subsample const &op,
                           Tensor &output,
                           Tensor const &input)
  {
    // Argument sanity check.
    assert(output) ;
    assert(input) ;
    TensorShape outShape ;
    op.forwardShape(outShape, input) ;
    assert(outShape == output) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int width = input.getWidth() ;
    Int height = input.getHeight() ;
    Int depth = input.getNumChannels() ;
    Int size = input.getCardinality() ;
    auto inputData = (type*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;

    Int outputHeight = output.getHeight() ;
    Int outputWidth = output.getWidth() ;
    Int strideY = op.getStride(0) ;
    Int strideX = op.getStride(1) ;
    Int padTop = op.getPadding(0) ;
    Int padLeft = op.getPadding(2) ;

    for (Int z = 0; z < depth * size ; ++z) {
      for (Int x = 0; x < outputWidth ; ++x) {
        for (Int y = 0; y < outputHeight ; ++y) {
          auto x1 = x * strideX - padLeft ;
          auto y1 = y * strideY - padTop ;
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
  vl::ErrorCode operator()(Subsample const &op,
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

    auto numOutputPixels = output.getHeight() * output.getWidth() ;
    type const* allOnesMemory = (type*) op.getContext().getAllOnes(deviceType, dataType, (size_t)numOutputPixels) ;

    if (allOnesMemory == NULL) {
      error = op.getContext().getLastError() ;
      goto done ;
    }

    for (Int image = 0 ; image < input.getCardinality() ; ++image) {
      auto outputOffset = (output.getHeight()*output.getWidth()*output.getNumChannels()) * image ;
      if (biases) {
        type alpha = 1 ;
        type beta = 1 ;
        error = vl::impl::blas<deviceType, dataType>::gemm
        (op.getContext(),
         'n', 'n',
         numOutputPixels,
         biases.getNumElements(), 1,
         alpha,
         allOnesMemory, numOutputPixels,
         (type*)biases.getMemory(), 1,
         beta,
         (type*)output.getMemory() + outputOffset, numOutputPixels) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }
  done:
    return op.getContext().passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct SubsampleBackward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(Subsample const &op,
                           Tensor &derInput,
                           Tensor const &derOutput)
  {
    assert(derInput) ;
    assert(derOutput) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = derInput.getWidth() ;
    auto height = derInput.getHeight() ;
    auto depth = derInput.getNumChannels() ;
    auto size = derInput.getCardinality() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derOutputData = (type*)derOutput.getMemory() ;

    // Check argument compatibility
    TensorShape outShape ;
    op.forwardShape(outShape, derInput) ;
    assert(outShape == derOutput) ;
    Int outputHeight = derOutput.getHeight() ;
    Int outputWidth = derOutput.getWidth() ;
    Int strideY = op.getStride(0) ;
    Int strideX = op.getStride(1) ;
    Int padTop = op.getPadding(0) ;
    Int padLeft = op.getPadding(2) ;

    memset(derInputData, 0, sizeof(type) * size_t(width * height * depth * size)) ;

    for (Int z = 0; z < depth * size; ++z) {
      for (Int px = 0; px < outputWidth; ++px) {
        for (Int py  = 0; py < outputHeight; ++py) {
          auto x1 = px * strideX - padLeft ;
          auto y1 = py * strideY - padTop ;
          if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            derInputData[x1 * height + y1]
            = derOutputData[px * outputHeight + py] ;
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
  vl::ErrorCode operator()(vl::nn::Subsample const &op,
                           vl::Tensor derInput,
                           vl::Tensor derBiases,
                           vl::Tensor derOutput)
  {
    assert(derOutput) ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    // Compute derInput.
    if (derInput) {
      error = SubsampleBackward<deviceType,dataType>()(op,derInput,derOutput) ;
      if (error != VLE_Success) { return error ; }
    }

    // Compute derBiases.
    if (derBiases) {
      auto numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
      type const* allOnesMemory = (type*) op.getContext().getAllOnes(deviceType, dataType, (size_t)numOutputPixels) ;

      if (allOnesMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }

      for (Int image = 0 ; image < derInput.getCardinality() ; ++image) {
        auto derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getNumChannels()) * image ;
        type alpha = 1 ;
        type beta = (image > 0) ; // Avoids having to clear derOutputs first.
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.getContext(),
         't',
         numOutputPixels, derOutput.getNumChannels(),
         alpha,
         (type const*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
         allOnesMemory, 1,
         beta,
         (type*)derBiases.getMemory(), 1) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }

  done:
    return op.getContext().passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_GPU
#include "nnsubsample_gpu.cu"
#endif

Subsample::Subsample(vl::Context &context,
                     Int strideY, Int strideX,
                     Int padTop, Int padBottom,
                     Int padLeft, Int padRight)
: ConvolutionLike(context,2)
{
  setStride({strideY, strideX}) ;
  setPadding({padTop, padBottom, padLeft, padRight}) ;
}

vl::ErrorCode
Subsample::forwardWithBias(vl::Tensor &output,
                           vl::Tensor const &input,
                           vl::Tensor const &biases) const
{
  return dispatch<SubsampleAndBiasForward>()(*this,output,input,biases) ;
}

vl::ErrorCode
Subsample::forwardShape(vl::TensorShape &output, vl::TensorShape const& input) const
{
  output = TensorShape() ; // null
  if (input.getNumDimensions() < getNumSpatialDimensions()) {
    return VLE_IllegalArgument ;
  }
  output = input ;
  for (Int d = 0 ; d < getNumSpatialDimensions() ; ++d) {
    auto odim = convLikeSizeHelper(input.getDimension(d),
                                   1,
                                   getStride(d),
                                   {getPadding(2*d),getPadding(2*d+1)},
                                   1) ;
    output.setDimension(d, odim) ;
  }
  return VLE_Success ;
}

vl::ErrorCode
Subsample::backwardWithBias(vl::Tensor &derInput,
                            vl::Tensor &derBiases,
                            vl::Tensor const &derOutput) const
{
  return dispatch<SubsampleAndBiasBackward>()(*this,derInput,derBiases,derOutput) ;
}
