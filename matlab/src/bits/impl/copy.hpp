// @file copy.hpp
// @brief Copy and other data operations
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__copy__
#define __vl__copy__

#include "../data.hpp"

namespace vl { namespace impl {

  template <vl::DeviceType dev, typename type>
  struct operations
  {
    static vl::ErrorCode copy(type * dst, type const * src, size_t numElements, double mult = 1.0) ;
    static vl::ErrorCode fill(type * dst, size_t numElements, type value) ;
  } ;
} }

#endif /* defined(__vl__copy__) */
