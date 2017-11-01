// @file nnbias.hpp
// @brief Bias block
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbias__
#define __vl__nnbias__

#include "nnoperation.hpp"

namespace vl { namespace nn {

  class Bias : public Operation {
  public:
    Bias(Context &context) ;

    ErrorCode forwardShape(TensorShape &output, TensorShape const& input) {
      output = input ;
      return VLE_Success ;
    }

    /// Compute output <- outputMult * output + inputMult * input + biasMult * bias.
    /// input can be empty, in which case it is dropped from the calculation.
    ErrorCode forward(Tensor &output, double outputMult,
                      Tensor const &input, double inputMult,
                      Tensor const &bias, double biasMult) ;

    // derInput and derBias can be empty to skip the calculation of the
    // corresponding derivative.
    ErrorCode backward(Tensor &derInput, double derInputMult,
                       Tensor &derBias, double derBiasMult,
                       double inputMult, double biasMult,
                       Tensor const &derOutput) ;
  } ;
  
} }

#endif /* defined(__vl__nnbias__) */
