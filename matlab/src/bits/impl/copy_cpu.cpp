//
//  copy.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 08/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "copy.hpp"
#include <string.h>

using namespace vl ;
using namespace vl::impl ;

template <> vl::Error
vl::impl::copy<vl::CPU, float>(float * dest,
                               float const * src,
                               size_t numElements)
{
  memcpy(dest, src, numElements * sizeof(float)) ;
  return vlSuccess ;
}

