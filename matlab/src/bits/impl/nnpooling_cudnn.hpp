// @file nnpooling_blas.hpp
// @brief Pooling block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnpooling_cudnn__
#define __vl__nnpooling_cudnn__

#include "../nnpooling.hpp"
#include "../data.hpp"
#include "cudnn.h"


namespace vl { namespace impl {

  template<typename type> vl::Error
  nnpooling_forward_cudnn(Context& context,
                          Tensor output,
                          Tensor data,
                          vl::PoolingMethod method,
                          int poolHeight, int poolWidth,
                          int strideY, int strideX,
                          int padTop, int padBottom,
                          int padLeft, int padRight) ;

  template<typename type> vl::Error
  nnpooling_backward_cudnn(Context& context,
                           Tensor derData,
                           Tensor data,
                           Tensor output,
                           Tensor derOutput,
                           vl::PoolingMethod method,
                           int poolHeight, int poolWidth,
                           int strideY, int strideX,
                           int padTop, int padBottom,
                           int padLeft, int padRight) ;

  /* specialisations */

  template<> vl::Error
  nnpooling_forward_cudnn<float>(Context& context,
                                 Tensor output,
                                 Tensor data,
                                 vl::PoolingMethod method,
                                 int poolHeight, int poolWidth,
                                 int strideY, int strideX,
                                 int padTop, int padBottom,
                                 int padLeft, int padRight) ;

  template<> vl::Error
  nnpooling_backward_cudnn<float>(Context& context,
                                  Tensor derData,
                                  Tensor data,
                                  Tensor output,
                                  Tensor derOutput,
                                  vl::PoolingMethod method,
                                  int poolHeight, int poolWidth,
                                  int strideY, int strideX,
                                  int padTop, int padBottom,
                                  int padLeft, int padRight) ;
} }

#endif /* defined(__vl__nnpooling_cudnn__) */
