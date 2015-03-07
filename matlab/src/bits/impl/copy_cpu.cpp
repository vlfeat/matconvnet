// @file copy_cpu.cpp
// @brief Copy data (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

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

