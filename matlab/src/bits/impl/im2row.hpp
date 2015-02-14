// @file im2row.hpp
// @brief Stack image patches as matrix rows
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__im2row__
#define __vl__im2row__

#include "../data.hpp"
#include <stddef.h>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  im2row(vl::Context& context,
         type* stacked,
         type const* data,
         size_t height, size_t width, size_t depth,
         size_t windowHeight, size_t windowWidth,
         size_t strideY, size_t strideX,
         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<vl::Device dev, typename type> vl::Error
  row2im(vl::Context& context,
         type* data,
         type const* stacked,
         size_t height, size_t width, size_t depth,
         size_t windowHeight, size_t windowWidth,
         size_t strideY, size_t strideX,
         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;


  /* Specializations */

  template<> vl::Error
  im2row<vl::CPU, float>(vl::Context& context,
                         float* stacked,
                         float const* data,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  row2im<vl::CPU, float>(vl::Context& context,
                         float* data,
                         float const* stacked,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

#if ENABLE_GPU
  template<> vl::Error
  im2row<vl::GPU, float>(vl::Context& context,
                         float* stacked,
                         float const* data,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<> vl::Error
  row2im<vl::GPU, float>(vl::Context& context,
                         float* data,
                         float const* stacked,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
#endif

} }

#endif /* defined(__vl__im2row__) */
