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

#include "data.hpp"

namespace vl { namespace nn {

  class Subsample {
  public:
    Subsample(vl::Context &context,
              int strideY, int strideX,
              int padTop, int padBottom,
              int padLeft, int padRight) ;

    vl::ErrorCode forwardWithBias(vl::Tensor &output,
                                  vl::Tensor const &input,
                                  vl::Tensor const &biases) ;

    vl::ErrorCode backwardWithBias(vl::Tensor &derInput,
                                   vl::Tensor &derBiases,
                                   vl::Tensor const &derOutput) ;

    vl::Context& context ;
    int strideY ;
    int strideX ;
    int padTop ;
    int padBottom ;
    int padLeft ;
    int padRight ;
  } ;

} }

#endif /* defined(__vl__nnsubsample__) */
