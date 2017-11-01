// @file nnbnorm.hpp
// @brief Batch normalizatoion block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Sebastien Ehrhardt and Andrea Vedaldi.
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbnorm__
#define __vl__nnbnorm__

#include "nnoperation.hpp"
#include <stdio.h>

namespace vl { namespace nn {

  class BatchNorm : public Operation {
  public:
    BatchNorm(vl::Context &context) ;
    BatchNorm(vl::Context &context, double epsilon) ;

    /// moment can be null
    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor &moment,
                          vl::Tensor const &input,
                          vl::Tensor const &multiplier,
                          vl::Tensor const &bias) const ;

    vl::ErrorCode forwardShape(vl::TensorShape &output,
                               vl::TensorShape &moments,
                               vl::TensorShape const &input) const ;

    vl::ErrorCode forwardWithMoment(vl::Tensor &output,
                                    vl::Tensor const &moment,
                                    vl::Tensor const &input,
                                    vl::Tensor const &multiplier,
                                    vl::Tensor const &bias) const ;

    /// moment can be null
    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor &derMultiplier,
                           vl::Tensor &derBias,
                           vl::Tensor &moment,
                           vl::Tensor const &input,
                           vl::Tensor const &multiplier,
                           vl::Tensor const &bias,
                           vl::Tensor const &derOutput) const ;

    vl::ErrorCode backwardWithMoment(vl::Tensor &derInput,
                                     vl::Tensor &derMultiplier,
                                     vl::Tensor &derBias,
                                     vl::Tensor const &moment,
                                     vl::Tensor const &input,
                                     vl::Tensor const &multiplier,
                                     vl::Tensor const &bias,
                                     vl::Tensor const &derOutput) const ;

    vl::ErrorCode setEpsilon(double epsilon)  ;
    double getEpsilon() const { return epsilon ; }

  private:
    double epsilon ;
  } ;

} }

#endif /* defined(__vl__nnbnorm__) */
