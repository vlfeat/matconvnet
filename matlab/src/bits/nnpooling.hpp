// @file nnpooling.hpp
// @brief Pooling layer.
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnpooling__
#define __vl__nnpooling__

#include "nnoperation.hpp"
#include <array>
#include <cassert>

namespace vl { namespace nn {

  class Pooling : public ConvolutionLike {
  public:
    enum Method { Max, Average } ;

    Pooling(vl::Context &context,
            Int poolHeight, Int poolWidth,
            Int strideY, Int strideX,
            Int padTop, Int padBottom,
            Int padLeft, Int padRight,
            Method method) ;

    Pooling(vl::Context &context) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &input) const ;

    vl::ErrorCode forwardShape(vl::TensorShape &output,
                               vl::TensorShape const &input) const ;

    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor const &input,
                           vl::Tensor const &derOutput) const ;

    vl::ErrorCode setShape(std::vector<Int> const& shape) ;

    Int getShape(Int index) const {
      assert(0 <= index && index < getNumSpatialDimensions()) ;
      return shape[as_unsigned(index)] ;
    }

    std::vector<Int> const& getShape() const { return shape ; }

    vl::ErrorCode setMethod(Method method) ;

    Method getMethod() const { return method ; }

  private:
    std::vector<Int> shape ;
    Method method ;
  } ;
  
} }

#endif /* defined(__vl__nnpooling__) */
