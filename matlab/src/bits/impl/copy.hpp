//
//  copy.h
//  matconv
//
//  Created by Andrea Vedaldi on 08/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __vl__copy__
#define __vl__copy__

#include "../data.hpp"

namespace vl { namespace impl {

  template <vl::Device dev, typename type> vl::Error
  copy(type * dest,
       type const * src,
       size_t numElements) ;

  template<> vl::Error
  copy<vl::CPU, float> (float * dest,
                        float const * src,
                        size_t numElements) ;

#if ENABLE_GPU
  template<> vl::Error
  copy<vl::GPU, float> (float * dest,
                        float const * src,
                        size_t numElements) ;
#endif

} }

/* ---------------------------------------------------------------- */
/* Implementation                                                   */
/* ---------------------------------------------------------------- */




#endif /* defined(__vl__copy__) */
