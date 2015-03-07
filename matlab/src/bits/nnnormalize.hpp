// @file nnnormalize.hpp
// @brief Normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/
#ifndef __vl__nnnormalize__
#define __vl__nnnormalize__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  vl::Error
  nnnormalize_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      size_t normDetph,
                      double kappa, double alpha, double beta) ;

  vl::Error
  nnnormalize_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor data,
                       vl::Tensor derOutput,
                       size_t normDetph,
                       double kappa, double alpha, double beta) ;
}

#endif /* defined(__vl__nnnormalize__) */
