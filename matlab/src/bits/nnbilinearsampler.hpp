// @file nnbilinearsampler.hpp
// @brief Bilinear sampler block
// @author Ankush Gupta
// @author Andrea Vedaldi

/*
Copyright (C) 2016-17 Ankush Gupta and Andrea Vedaldi.
All rights reserved.
This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbilinearsampler__
#define __vl__nnbilinearsampler__

#include "data.hpp"
#include <stdio.h>

namespace vl { namespace nn {

  class BilinearSampler {
  public:
    BilinearSampler(Context &context) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &input,
                          vl::Tensor const &grid) ;

    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor &derGrid,
                           vl::Tensor const &input,
                           vl::Tensor const &grid,
                           vl::Tensor const &derOutput) ;

    Context &context ;
  } ;

} }

#endif /* defined(__vl__nnbilinearsampler__) */
