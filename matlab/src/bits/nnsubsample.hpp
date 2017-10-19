// @file nnsubsample.hpp
// @brief Subsamping block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnsubsample__
#define __vl__nnsubsample__

#include "nnoperation.hpp"
#include <array>
#include <cassert>

namespace vl { namespace nn {

  class Subsample : public ConvolutionLike {
  public:
    Subsample(vl::Context &context,
              Int strideY, Int strideX,
              Int padTop, Int padBottom,
              Int padLeft, Int padRight) ;

    vl::ErrorCode forwardWithBias(vl::Tensor &output,
                                  vl::Tensor const &input,
                                  vl::Tensor const &biases) const ;

    vl::ErrorCode forwardShape(vl::TensorShape &output, vl::TensorShape const& input) const ;

    vl::ErrorCode backwardWithBias(vl::Tensor &derInput,
                                   vl::Tensor &derBiases,
                                   vl::Tensor const &derOutput) const ;
  } ;

} }

#endif /* defined(__vl__nnsubsample__) */
