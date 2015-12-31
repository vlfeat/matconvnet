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

#ifndef VL_POOLING_H
#define VL_POOLING_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::Device dev, typename type>
  struct pooling_max {

    static vl::Error
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

    static vl::Error
    backward(type* derData,
             type const* data,
             type const* derPooled,
             size_t height, size_t width, size_t depth,
             size_t poolHeight, size_t poolWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
  } ;

  template<vl::Device dev, typename type>
  struct pooling_average {

    static vl::Error
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

    static vl::Error
    backward(type* derData,
             type const* derPooled,
             size_t height, size_t width, size_t depth,
             size_t poolHeight, size_t poolWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
  } ;

} }

#endif /* defined(VL_POOLING_H) */
