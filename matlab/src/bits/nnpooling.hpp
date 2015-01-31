//
//  nnpooling.h
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__nnpooling__
#define __matconv__nnpooling__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  enum PoolingMethod { MAX, AVERAGE } ;

  int nnpooling_forward(vl::Context& context,
                        vl::Tensor output,
                        vl::Tensor data,
                        PoolingMethod method,
                        int poolHeight, int poolWidth,
                        int strideY, int strideX,
                        int padTop, int padBottom,
                        int padLeft, int padRight) ;

  int nnpooling_backward(vl::Context& context,
                         vl::Tensor derData,
                         vl::Tensor data,
                         vl::Tensor derOutput,
                         PoolingMethod method,
                         int poolHeight, int poolWidth,
                         int strideY, int strideX,
                         int padTop, int padBottom,
                         int padLeft, int padRight) ;
}

#endif /* defined(__matconv__nnpooling__) */
