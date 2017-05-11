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

#include "data.hpp"

namespace vl { namespace nn {

  class Bias {
  public:
    Bias(vl::Context &context) ;

    // output <- outputMult * output + inputMult * input + biasMult * bias
    vl::ErrorCode forward(vl::Tensor &output, double outputMult,
                          vl::Tensor const &input, double inputMult,
                          vl::Tensor const &bias, double biasMult) ;

    vl::ErrorCode backward(vl::Tensor &derInput, double derInputMult,
                           vl::Tensor &derBias, double derBiasMult,
                           double inputMult, double biasMult,
                           vl::Tensor const &derOutput) ;

    vl::Context& context ;
  } ;
  
} }

#endif /* defined(__vl__nnbias__) */
