//
//  nnsubsample.h
//  matconv
//
//  Created by Andrea Vedaldi on 07/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__nnsubsample__
#define __matconv__nnsubsample__

#include "data.hpp"

namespace vl {

  int nnsubsample_forward(vl::Context& context,
                          vl::Tensor output,
                          vl::Tensor data,
                          vl::Tensor biases,
                          int strideY, int strideX,
                          int padTop, int padBottom,
                          int padLeft, int padRight) ;

  int nnsubsample_backward(vl::Context& context,
                           vl::Tensor derData,
                           vl::Tensor derBiases,
                           vl::Tensor derOutput,
                           int strideY, int strideX,
                           int padTop, int padBottom,
                           int padLeft, int padRight) ;
}

#endif /* defined(__matconv__nnsubsample__) */
