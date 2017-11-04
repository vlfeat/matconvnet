// @file nnnormalizelp.hpp
// @brief Lp normalization block
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
    NormalizeLp(Context &context) ;
    NormalizeLp(Context &context,
                std::vector<Int> const& selectedDimensions,
                double exponent = 2.0,
                double epsilon = 1e-3) ;

    ErrorCode forward(Tensor &output,
                      Tensor &norms,
                      Tensor const &data) const ;

    ErrorCode forwardShape(TensorShape &output,
                           TensorShape &norms,
                           TensorShape const &data) const ;

    ErrorCode forwardWithNorms(Tensor &output,
                               Tensor const &norms,
                               Tensor const &data) const ;

    ErrorCode backward(Tensor &derData,
                       Tensor &moments,
                       Tensor const &data,
                       Tensor const &derOutput) const ;

    ErrorCode backwardWithNorms(Tensor &derData,
                                Tensor const &norms,
                                Tensor const &data,
                                Tensor const &derOutput) const ;

    double getExponent() const { return exponent ; }
    ErrorCode setExponent(double exponent) {
      this->exponent = exponent ;
      return VLE_Success ;
    }

    double getEpsilon() const { return epsilon ; }
    ErrorCode setEpsilon(double epsilon) {
      this->epsilon = epsilon ;
      return VLE_Success ;
    }

    std::vector<Int> const& getSelectedDimensions() const {
      return selectedDimensions ;
    }
    ErrorCode setSelectedDimensions(std::vector<Int> const&) ;

  private:
    std::vector<Int> selectedDimensions ;
    double exponent ;
    double epsilon ;
  } ;

} }

#endif /* defined(__nnnormalizelp__) */
