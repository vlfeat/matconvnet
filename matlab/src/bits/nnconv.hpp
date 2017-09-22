// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015-17 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv__
#define __vl__nnconv__

#include "data.hpp"

namespace vl { namespace nn {

  class Convolution {
  public:
    Convolution(Context &context,
                int strideY, int strideX,
                int padTop, int padBottom,
                int padLeft, int padRight,
                int dilateY, int dilateX) ;

    vl::ErrorCode forward(vl::Tensor &output, double outputMult,
                          vl::Tensor const& input, double inputMult,
                          vl::Tensor const& filter,
                          vl::Tensor const& bias) ;

    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor &derFilter,
                           vl::Tensor &derBias,
                           vl::Tensor const &input,
                           vl::Tensor const &filter,
                           vl::Tensor const &derOutput) ;

    Context &context ;
    int strideY ;
    int strideX ;
    int padTop ;
    int padBottom ;
    int padLeft ;
    int padRight ;
    int dilateY ;
    int dilateX ;
  } ;

  class ConvolutionTranspose {
  public:
    ConvolutionTranspose(Context &context,
                         int upsampleY, int upsampleX,
                         int cropTop, int cropBottom,
                         int cropLeft, int cropRight) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &input,
                          vl::Tensor const &filter,
                          vl::Tensor const &bias) ;

    vl::ErrorCode backward(vl::Tensor &derData,
                           vl::Tensor &derFilter,
                           vl::Tensor &derBias,
                           vl::Tensor const &input,
                           vl::Tensor const &filter,
                           vl::Tensor const &derOutput);

    Context &context ;
    int upsampleY ;
    int upsampleX ;
    int cropTop ;
    int cropBottom ;
    int cropLeft ;
    int cropRight ;
  } ;

} }


#endif /* defined(__vl__nnconv__) */
