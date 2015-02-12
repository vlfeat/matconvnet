//
//  nnsubsample.h
//  matconv
//
//  Created by Andrea Vedaldi on 07/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __vl__nnsubsample__
#define __vl__nnsubsample__

#include "data.hpp"

namespace vl {

  vl::Error
  nnsubsample_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      vl::Tensor biases,
                      int strideY, int strideX,
                      int padTop, int padBottom,
                      int padLeft, int padRight) ;

  vl::Error
  nnsubsample_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor derBiases,
                       vl::Tensor derOutput,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight) ;
}

#endif /* defined(__vl__nnsubsample__) */
