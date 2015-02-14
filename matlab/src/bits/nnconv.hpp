// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014-15 Andrea Vedaldi and Max Jaderberg.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv__
#define __vl__nnconv__

#include "data.hpp"

namespace vl {

  vl::Error
  nnconv_forward(vl::Context& context,
                 vl::Tensor output,
                 vl::Tensor data,
                 vl::Tensor filters,
                 vl::Tensor biases,
                 int strideY, int strideX,
                 int padTop, int padBottom,
                 int padLeft, int padRight) ;

  vl::Error
  nnconv_backward(vl::Context& context,
                  vl::Tensor derData,
                  vl::Tensor derFilters,
                  vl::Tensor derBiases,
                  vl::Tensor data,
                  vl::Tensor filters,
                  vl::Tensor derOutput,
                  int strideY, int strideX,
                  int padTop, int padBottom,
                  int padLeft, int padRight) ;
}


#endif /* defined(__vl__nnconv__) */
