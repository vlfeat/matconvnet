// @file subsampling.hpp
// @brief Subsampling block implementation
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_NNSUBSAMPLE_H
#define VL_NNSUBSAMPLE_H

#include "../data.hpp"
#include <stddef.h>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  subsample_forward(vl::Context& context,
                    type* subsampled,
                    type const* data,
                    size_t height, size_t width, size_t depth,
                    size_t strideY, size_t strideX,
                    size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<vl::Device dev, typename type> vl::Error
  subsample_backward(vl::Context& context,
                     type* derData,
                     type const* derSubsampled,
                     size_t height, size_t width, size_t depth,
                     size_t strideY, size_t strideX,
                     size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  /* Specializations */

  template<> vl::Error
  subsample_forward<vl::CPU, float>(vl::Context& context,
                                    float* subsampled,
                                    float const* data,
                                    size_t height, size_t width, size_t depth,
                                    size_t strideY, size_t strideX,
                                    size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  subsample_backward<vl::CPU, float>(vl::Context& context,
                                     float* derData,
                                     float const* derSubsampled,
                                     size_t height, size_t width, size_t depth,
                                     size_t strideY, size_t strideX,
                                     size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

#if ENABLE_GPU
  template<> vl::Error
  subsample_forward<vl::GPU, float>(vl::Context& context,
                                    float* stacked,
                                    float const* data,
                                    size_t height, size_t width, size_t depth,
                                    size_t strideY, size_t strideX,
                                    size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  subsample_backward<vl::GPU, float>(vl::Context& context,
                                     float* derData,
                                     float const* derSubsampled,
                                     size_t height, size_t width, size_t depth,
                                     size_t strideY, size_t strideX,
                                     size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
#endif

} }

#endif /* defined(VL_NNSUBSAMPLE_H) */
