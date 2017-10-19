// @file nnnormalizelp.hpp
// @brief Batch normalizatoion block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __nnnormalizelp__
#define __nnnormalizelp__

#include "nnoperation.hpp"
#include <vector>

namespace vl { namespace nn {

  class NormalizeLp : public Operation {
  public:
    NormalizeLp(vl::Context &context,
                std::vector<Int> const& selectedDimensions,
                double exponent = 2.0,
                double epsilon = 1e-3) ;

    vl::TensorShape getNormsShapeForData(vl::Tensor const &data) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor &norms,
                          vl::Tensor const &data) ;

    vl::ErrorCode forwardWithNorms(vl::Tensor &output,
                                   vl::Tensor const &norms,
                                   vl::Tensor const &data) ;

    vl::ErrorCode backward(vl::Tensor &derData,
                           vl::Tensor &moments,
                           vl::Tensor const &data,
                           vl::Tensor const &derOutput) ;

    vl::ErrorCode backwardWithNorms(vl::Tensor &derData,
                                    vl::Tensor const &norms,
                                    vl::Tensor const &data,
                                    vl::Tensor const &derOutput) ;


    double getExponent() const { return exponent ; }
    double getEpsilon() const { return epsilon ; }
    std::vector<Int> const& getSelectedDimensions() const {
      return selectedDimensions ;
    }

  private:
    std::vector<Int> selectedDimensions ;
    double exponent ;
    double epsilon ;
  } ;

} }

#endif /* defined(__nnnormalizelp__) */
