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
#include "nnoperation.hpp"
#include <stdio.h>

namespace vl { namespace nn {

  class BilinearSampler : public Operation {
  public:
    BilinearSampler(Context &context) ;

    ErrorCode forward(Tensor &output,
                      Tensor const &input,
                      Tensor const &grid) ;

    ErrorCode forwardShape(TensorShape &output,
                           TensorShape const &input,
                           TensorShape const &grid) ;

    ErrorCode backward(Tensor &derInput,
                       Tensor &derGrid,
                       Tensor const &input,
                       Tensor const &grid,
                       Tensor const &derOutput) ;
  } ;

} }

#endif /* defined(__vl__nnbilinearsampler__) */
