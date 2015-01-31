/** @file im2row.hpp
 ** @brief Extracts feature map patches as rows of a matrix and back
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __matconv__im2row__
#define __matconv__im2row__

#include "../data.hpp"
#include <stddef.h>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> int
  im2row(vl::Context& context,
         type* stacked,
         type const* data,
         size_t height, size_t width, size_t depth,
         size_t windowHeight, size_t windowWidth,
         size_t strideY, size_t strideX,
         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template<vl::Device dev, typename type> int
  row2im(vl::Context& context,
         type* data,
         type const* stacked,
         size_t height, size_t width, size_t depth,
         size_t windowHeight, size_t windowWidth,
         size_t strideY, size_t strideX,
         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;


  /* Specializations */

  template <> int
  im2row<vl::CPU, float>(vl::Context& context,
                         float* stacked,
                         float const* data,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template <> int
  row2im<vl::CPU, float>(vl::Context& context,
                         float* data,
                         float const* stacked,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

#if ENABLE_GPU
  template <> int
  im2row<vl::GPU, float>(vl::Context& context,
                         float* stacked,
                         float const* data,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

  template <> int
  row2im<vl::GPU, float>(vl::Context& context,
                         float* data,
                         float const* stacked,
                         size_t height, size_t width, size_t depth,
                         size_t windowHeight, size_t windowWidth,
                         size_t strideY, size_t strideX,
                         size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
#endif

} }

#endif /* defined(__matconv__im2row__) */
