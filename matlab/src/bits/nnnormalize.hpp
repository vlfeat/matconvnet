// @file nnnormalize.hpp
// @brief Normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnnormalize__
#define __vl__nnnormalize__

#include "data.hpp"
#include <stdio.h>

namespace vl { namespace nn {

  class LRN {
  public:
    LRN(vl::Context &context,
        int normDepth = 5,
        double kappa = 2.0,
        double alpha = 1e-3,
        double beta = 0.5) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &data) ;

    vl::ErrorCode backward(vl::Tensor &derData,
                           vl::Tensor const &data,
                           vl::Tensor const &derOutput) ;
    vl::Context& context ;
    double kappa ;
    double alpha ;
    double beta ;
    int normDepth ;
  } ;

} }

#endif /* defined(__vl__nnnormalize__) */
