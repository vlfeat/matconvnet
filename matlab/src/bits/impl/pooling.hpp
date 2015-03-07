// @file pooling.hpp
// @brief Pooling block implementation
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_NNPOOLING_H
#define VL_NNPOOLING_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  /* Max pooling */

  template<vl::Device dev, typename type> vl::Error
  pooling_max_forward(type* pooled,
                      type const* data,
                      size_t height, size_t width, size_t depth,
                      size_t poolHeight, size_t poolWidth,
                      size_t strideY, size_t strideX,
                      size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<vl::Device dev, typename type> vl::Error
  pooling_max_backward(type* derData,
                       type const* data,
                       type const* derPooled,
                       size_t height, size_t width, size_t depth,
                       size_t poolHeight, size_t poolWidth,
                       size_t strideY, size_t strideX,
                       size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  /* Sum pooling */
  template<vl::Device dev, typename type> vl::Error
  pooling_average_forward(type* pooled,
                          type const* data,
                          size_t height, size_t width, size_t depth,
                          size_t poolHeight, size_t poolWidth,
                          size_t strideY, size_t strideX,
                          size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<vl::Device dev, typename type> vl::Error
  pooling_average_backward(type* derData,
                           type const* derPooled,
                           size_t height, size_t width, size_t depth,
                           size_t poolHeight, size_t poolWidth,
                           size_t strideY, size_t strideX,
                               size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  /* Specializations: CPU, float */

  template<> vl::Error
  pooling_max_forward<vl::CPU, float>(float* pooled,
                                      float const* data,
                                      size_t height, size_t width, size_t depth,
                                      size_t poolHeight, size_t poolWidth,
                                      size_t strideY, size_t strideX,
                                      size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  pooling_max_backward<vl::CPU, float>(float* derData,
                                       float const* data,
                                       float const* derPooled,
                                       size_t height, size_t width, size_t depth,
                                       size_t poolHeight, size_t poolWidth,
                                       size_t strideY, size_t strideX,
                                       size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  pooling_average_forward<vl::CPU, float>(float* pooled,
                                          float const* data,
                                          size_t height, size_t width, size_t depth,
                                          size_t poolHeight, size_t poolWidth,
                                          size_t strideY, size_t strideX,
                                          size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  pooling_average_backward<vl::CPU, float>(float* derData,
                                           float const* derPooled,
                                           size_t height, size_t width, size_t depth,
                                           size_t poolHeight, size_t poolWidth,
                                           size_t strideY, size_t strideX,
                                           size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  /* Specializations: GPU, float */

#if ENABLE_GPU
  template<> vl::Error
  pooling_max_forward<vl::GPU, float>(float* pooled,
                                      float const* data,
                                      size_t height, size_t width, size_t depth,
                                      size_t poolHeight, size_t poolWidth,
                                      size_t strideY, size_t strideX,
                                      size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  pooling_max_backward<vl::GPU, float>(float* derData,
                                       float const* data,
                                       float const* derPooled,
                                       size_t height, size_t width, size_t depth,
                                       size_t poolHeight, size_t poolWidth,
                                       size_t strideY, size_t strideX,
                                       size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  pooling_average_forward<vl::GPU, float>(float* pooled,
                                          float const* data,
                                          size_t height, size_t width, size_t depth,
                                          size_t poolHeight, size_t poolWidth,
                                          size_t strideY, size_t strideX,
                                          size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  pooling_average_backward<vl::GPU, float>(float* derData,
                                           float const* derPooled,
                                           size_t height, size_t width, size_t depth,
                                           size_t poolHeight, size_t poolWidth,
                                           size_t strideY, size_t strideX,
                                           size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
#endif

} }

#endif /* defined(VL_NNPOOLING_H) */
