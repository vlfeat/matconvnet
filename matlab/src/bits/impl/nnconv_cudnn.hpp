// @file nnconv_blas.hpp
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv_cudnn__
#define __vl__nnconv_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<typename type> vl::Error
  nnconv_forward_cudnn(Context& context,
                       Tensor output, double outputMult,
                       Tensor data, double dataMult,
                       Tensor filters,
                       Tensor biases,
                       int strideX, int strideY,
                       int padLeft, int padRight,
                       int padTop, int padBottom) ;

  template<typename type> vl::Error
  nnconv_backward_cudnn(Context& context,
                        Tensor derData,
                        Tensor derFilters,
                        Tensor derBiases,
                        Tensor data,
                        Tensor filters,
                        Tensor derOutput,
                        int strideX, int strideY,
                        int padLeft, int padRight,
                        int padTop, int padBottom) ;

  /* specializations */

  template<> vl::Error
  nnconv_forward_cudnn<float>(Context& context,
                              Tensor output, double outputMult,
                              Tensor data, double dataMult,
                              Tensor filters,
                              Tensor biases,
                              int strideX, int strideY,
                              int padLeft, int padRight,
                              int padTop, int padBottom) ;

  template<> vl::Error
  nnconv_backward_cudnn<float>(Context& context,
                               Tensor derData,
                               Tensor derFilters,
                               Tensor derBiases,
                               Tensor data,
                               Tensor filters,
                               Tensor derOutput,
                               int strideX, int strideY,
                               int padLeft, int padRight,
                               int padTop, int padBottom) ;
} }

#endif /* defined(__vl__nnconv_cudnn__) */
