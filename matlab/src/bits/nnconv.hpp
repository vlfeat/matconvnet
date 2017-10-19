// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015-17 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv__
#define __vl__nnconv__

#include "nnoperation.hpp"
#include <array>
#include <cassert>

namespace vl { namespace nn {

  class Convolution : public ConvolutionLike {
  public:
    Convolution(Context &context,
                Int strideY, Int strideX,
                Int padTop, Int padBottom,
                Int padLeft, Int padRight,
                Int dilateY, Int dilateX) ;

    Convolution(Context &context) ;

    vl::ErrorCode forward(vl::Tensor &output, double outputMult,
                          vl::Tensor const& input, double inputMult,
                          vl::Tensor const& filter,
                          vl::Tensor const& bias) ;

    vl::ErrorCode forwardShape(vl::TensorShape &output,
                               vl::TensorShape const& input,
                               vl::TensorShape const& filter) ;

    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor &derFilter,
                           vl::Tensor &derBias,
                           vl::Tensor const &input,
                           vl::Tensor const &filter,
                           vl::Tensor const &derOutput) ;

    vl::ErrorCode backwardShape(vl::Tensor &output, double outputMult,
                                vl::Tensor const& input, double inputMult,
                                vl::Tensor const& filter,
                                vl::Tensor const& bias) ;

    vl::ErrorCode setDilation(std::vector<Int> const& dilation) ;

    Int getDilation(Int index) const {
      assert(0 <= index && index < as_signed(vl::Tensor::maxNumDimensions)) ;
      return dilation[as_unsigned(index)] ;
    }

  private:
    std::array<Int, vl::Tensor::maxNumDimensions> dilation ;
  } ;

  class ConvolutionTranspose : public Operation {
  public:
    ConvolutionTranspose(Context &context,
                         Int upsampleY, Int upsampleX,
                         Int cropTop, Int cropBottom,
                         Int cropLeft, Int cropRight) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &input,
                          vl::Tensor const &filter,
                          vl::Tensor const &bias) ;

    vl::ErrorCode backward(vl::Tensor &derData,
                           vl::Tensor &derFilter,
                           vl::Tensor &derBias,
                           vl::Tensor const &input,
                           vl::Tensor const &filter,
                           vl::Tensor const &derOutput);

    Int getCrop(Int index) const {
      assert(0 <= index && index < 2*as_signed(vl::Tensor::maxNumDimensions)) ;
      return crop[as_unsigned(index)] ;
    }

    Int getUpsample(Int index) const {
      assert(0 <= index && index < as_signed(vl::Tensor::maxNumDimensions)) ;
      return upsample[as_unsigned(index)] ;
    }

  private:
    Int numSpatialDimensions ;
    std::array<Int, vl::Tensor::maxNumDimensions> upsample ;
    std::array<Int, 2*vl::Tensor::maxNumDimensions> crop ;
  } ;

} }


#endif /* defined(__vl__nnconv__) */
