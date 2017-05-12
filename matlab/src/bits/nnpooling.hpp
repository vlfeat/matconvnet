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

#include "data.hpp"
#include <stdio.h>

namespace vl { namespace nn {

  class Pooling {
  public:
    enum Method { Max, Average } ;

    Pooling(vl::Context &context,
            int poolHeight, int poolWidth,
            int strideY, int strideX,
            int padTop, int padBottom,
            int padLeft, int padRight,
            Method method) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &input) ;

    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor const &input,
                           vl::Tensor const &derOutput) ;

    vl::Context& context ;
    int poolHeight ;
    int poolWidth ;
    int strideY ;
    int strideX ;
    int padTop ;
    int padBottom ;
    int padLeft ;
    int padRight ;
    Method method ;
  } ;
  
} }

#endif /* defined(__vl__nnpooling__) */
