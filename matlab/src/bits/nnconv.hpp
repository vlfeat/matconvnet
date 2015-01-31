/** @file nnconv.cu
 ** @brief Convolution block
 ** @author Andrea Vedaldi
 **/

/*
 Copyright (C) 2014-15 Andrea Vedaldi and Max Jaderberg.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#ifndef __matconv__nnconv__
#define __matconv__nnconv__

#include "data.hpp"

namespace vl {

  int nnconv_forward(vl::Context& context,
                     vl::Tensor output,
                     vl::Tensor data,
                     vl::Tensor filters,
                     vl::Tensor biases,
                     int strideY, int strideX,
                     int padTop, int padBottom,
                     int padLeft, int padRight) ;

  int nnconv_backward(vl::Context& context,
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


#endif /* defined(__matconv__nnconv__) */
