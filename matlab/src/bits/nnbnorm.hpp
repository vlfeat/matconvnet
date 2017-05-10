// @file nnbnorm.hpp
// @brief Batch normalizatoion block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
 Copyright (C) 2015-16 Sebastien Ehrhardt and Andrea Vedaldi.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#ifndef __vl__nnbnorm__
#define __vl__nnbnorm__

#include "data.hpp"
#include <stdio.h>

namespace vl { namespace nn {

  class BatchNorm {
  public:
    BatchNorm(vl::Context &context,
              double epsilon) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor &moment,
                          vl::Tensor const &input,
                          vl::Tensor const &multiplier,
                          vl::Tensor const &bias) ;

    vl::ErrorCode forwardWithMoment(vl::Tensor &output,
                                    vl::Tensor const &moment,
                                    vl::Tensor const &input,
                                    vl::Tensor const &multiplier,
                                    vl::Tensor const &bias) ;

    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor &derMultiplier,
                           vl::Tensor &derBias,
                           vl::Tensor &moment,
                           vl::Tensor const &input,
                           vl::Tensor const &multiplier,
                           vl::Tensor const &bias,
                           vl::Tensor const &derOutput) ;

    vl::ErrorCode backwardWithMoment(vl::Tensor &derInput,
                                     vl::Tensor &derMultiplier,
                                     vl::Tensor &derBias,
                                     vl::Tensor const &moment,
                                     vl::Tensor const &input,
                                     vl::Tensor const &multiplier,
                                     vl::Tensor const &bias,
                                     vl::Tensor const &derOutput) ;

    vl::Context& context ;
    double epsilon ;
  } ;

} }

#endif /* defined(__vl__nnbnorm__) */
