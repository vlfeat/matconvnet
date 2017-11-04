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
    Convolution(Context &context) ;
    Convolution(Context &context,
                Int strideY, Int strideX,
                Int padTop, Int padBottom,
                Int padLeft, Int padRight,
                Int dilateY, Int dilateX) ;

    ErrorCode forwardShape(TensorShape &output,
                           TensorShape const& input,
                           TensorShape const& filter) const ;

    ErrorCode forward(Tensor &output, double outputMult,
                      Tensor const& input, double inputMult,
                      Tensor const& filter,
                      Tensor const& bias) const ;

    ErrorCode backward(Tensor &derInput,
                       Tensor &derFilter,
                       Tensor &derBias,
                       Tensor const &input,
                       Tensor const &filter,
                       Tensor const &derOutput) const ;

    ErrorCode backwardShape(Tensor &output, double outputMult,
                            Tensor const& input, double inputMult,
                            Tensor const& filter,
                            Tensor const& bias) const ;

    ErrorCode setDilation(std::vector<Int> const& dilation) ;

    Int getDilation(Int index) const {
      assert(0 <= index && index < getNumSpatialDimensions()) ;
      return dilation[as_unsigned(index)] ;
    }

    std::vector<Int> const & getDilations() const {
      return dilation ;
    }

  private:
    std::vector<Int> dilation ;
  } ;

  class ConvolutionTranspose : public Operation {
  public:
    ConvolutionTranspose(Context &context) ;

    ConvolutionTranspose(Context &context,
                         Int upsampleY, Int upsampleX,
                         Int cropTop, Int cropBottom,
                         Int cropLeft, Int cropRight) ;

    ErrorCode forwardShape(TensorShape &output,
                           TensorShape const& input,
                           TensorShape const& filter) const ;

    ErrorCode forward(Tensor &output,
                      Tensor const &input,
                      Tensor const &filter,
                      Tensor const &bias) const ;

    ErrorCode backward(Tensor &derData,
                       Tensor &derFilter,
                       Tensor &derBias,
                       Tensor const &input,
                       Tensor const &filter,
                       Tensor const &derOutput) const ;

    Int getNumSpatialDimensions() const {
      return numSpatialDimensions ;
    }

    ErrorCode setNumFilterGroups(Int numFilterGroups) ;
    Int getNumFilterGroups() const { return numFilterGroups ; }
    ErrorCode setUpsampling(std::vector<Int> const& upsampling) ;
    ErrorCode setCropping(std::vector<Int> const& cropping) ;

    Int getCropping(Int index) const {
      assert(0 <= index && index < 2*as_signed(Tensor::maxNumDimensions)) ;
      return cropping[as_unsigned(index)] ;
    }

    Int getUpsampling(Int index) const {
      assert(0 <= index && index < as_signed(Tensor::maxNumDimensions)) ;
      return upsampling[as_unsigned(index)] ;
    }

    std::vector<Int> getCroppings() const {
      return {begin(cropping), begin(cropping)+2*getNumSpatialDimensions()} ;
    }

    std::vector<Int> getUpsamplings() const {
      return {begin(upsampling), begin(upsampling)+getNumSpatialDimensions()} ;
    }

  private:
    Int numSpatialDimensions ;
    Int numFilterGroups ;
    std::array<Int, Tensor::maxNumDimensions> upsampling ;
    std::array<Int, 2*Tensor::maxNumDimensions> cropping ;
  } ;

} }

#endif /* defined(__vl__nnconv__) */
