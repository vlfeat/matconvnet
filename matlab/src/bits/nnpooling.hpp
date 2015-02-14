//
//  nnpooling.h
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __vl__nnpooling__
#define __vl__nnpooling__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  enum PoolingMethod { vlPoolingMax, vlPoolingAverage } ;

  vl::Error
  nnpooling_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data,
                    PoolingMethod method,
                    int poolHeight, int poolWidth,
                    int strideY, int strideX,
                    int padTop, int padBottom,
                    int padLeft, int padRight) ;

  vl::Error
  nnpooling_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor derOutput,
                     PoolingMethod method,
                     int poolHeight, int poolWidth,
                     int strideY, int strideX,
                     int padTop, int padBottom,
                     int padLeft, int padRight) ;
}

#endif /* defined(__vl__nnpooling__) */
