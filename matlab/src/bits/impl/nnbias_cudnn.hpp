// @file nnbias_blas.hpp
// @brief biasolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbias_cudnn__
#define __vl__nnbias_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<typename type> vl::Error
  nnbias_forward_cudnn(vl::Context& context,
                       vl::Tensor output, double outputMult,
                       vl::Tensor data, double dataMult,
                       vl::Tensor biases, double biasesMult) ;

  template<typename type> vl::Error
  nnbias_backward_cudnn(vl::Context& context,
                        vl::Tensor derData, double derDataMult,
                        vl::Tensor derBiases, double derBiasesMult,
                        vl::Tensor derOutput, double derOutputMult) ;

  /* specializations */

  template<> vl::Error
  nnbias_forward_cudnn<float>(vl::Context& context,
                              vl::Tensor output, double outputMult,
                              vl::Tensor data, double dataMult,
                              vl::Tensor biases, double biasesMult) ;

  template<> vl::Error
  nnbias_backward_cudnn<float>(vl::Context& context,
                               vl::Tensor derData, double derDataMult,
                               vl::Tensor derBiases, double derBiasesMult,
                               vl::Tensor derOutput, double derOutputMult) ;
} }

#endif /* defined(__vl__nnbias_cudnn__) */
