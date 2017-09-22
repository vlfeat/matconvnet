// @file copy_cpu.cpp
// @brief Copy and other data operations (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "copy.hpp"
#include <string.h>

namespace vl { namespace impl {

  template <typename type>
  struct operations<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    copy(type * dst,
         type const * src,
         size_t numElements,
         double mult)
    {
      if (mult == 1.0) {
        memcpy(dst, src, numElements * sizeof(type)) ;
      } else {
        auto end = src + numElements ;
        while (src != end) {
          *dst++ = mult * (*src++) ;
        }
      }
      return VLE_Success ;
    }

    static vl::ErrorCode
    fill(type * dst,
         size_t numElements,
         type value)
    {
      for (size_t k = 0 ; k < numElements ; ++k) {
        dst[k] = value ;
      }
      return VLE_Success ;
    }
  } ;

} }

template struct vl::impl::operations<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::operations<vl::VLDT_CPU, double> ;
#endif


