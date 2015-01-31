//
//  copy.h
//  matconv
//
//  Created by Andrea Vedaldi on 08/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__copy__
#define __matconv__copy__

#include "../data.hpp"

namespace vl { namespace impl {

  template <vl::Device dev, typename type> inline int
  copy(type * dest,
       type const * src,
       size_t numElements) ;

  template <> int
  copy<vl::CPU, float> (float * dest,
                        float const * src,
                        size_t numElements) ;

#if ENABLE_GPU
  template <> int
  copy<vl::GPU, float> (float * dest,
                        float const * src,
                        size_t numElements) ;
#endif

} }

/* ---------------------------------------------------------------- */
/* Implementation                                                   */
/* ---------------------------------------------------------------- */




#endif /* defined(__matconv__copy__) */
