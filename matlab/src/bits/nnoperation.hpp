//
//  nnoperation.hpp
//  mcn-lib-cpu
//
//  Created by Andrea Vedaldi on 16/10/2017.
//  Copyright © 2017 Andrea Vedaldi. All rights reserved.
//

#ifndef nnoperation_hpp
#define nnoperation_hpp

#include "data.hpp"
#include <array>
#include <cassert>

namespace vl { namespace nn {

  class Operation {
  public:
    Operation(vl::Context& context)
    : context(context) { }

    Context& getContext() const { return context ; }

  private:
    Context& context ;
  } ;

  class ConvolutionLike : public Operation {
  public:
    ConvolutionLike(Context& context, Int numSpatialDimensions = 2) ;

    vl::ErrorCode setStride(std::vector<Int> const& stride) ;
    vl::ErrorCode setPadding(std::vector<Int> const& padding) ;

    Int getNumSpatialDimensions() const { return numSpatialDimensions ; }

    Int getPadding(Int index) const {
      assert(0 <= index && index < 2*as_signed(vl::Tensor::maxNumDimensions)) ;
      return padding[as_unsigned(index)] ;
    }

    Int getStride(Int index) const {
      assert(0 <= index && index < as_signed(vl::Tensor::maxNumDimensions)) ;
      return stride[as_unsigned(index)] ;
    }

    std::vector<Int> getStrides() const {
      return {begin(stride), begin(stride)+numSpatialDimensions} ;
    }

    std::vector<Int> getPaddings() const {
      return {begin(padding), begin(padding)+2*numSpatialDimensions} ;
    }

  private:
    Int numSpatialDimensions ;
    std::array<Int, vl::Tensor::maxNumDimensions> stride ;
    std::array<Int, 2*vl::Tensor::maxNumDimensions> padding ;
  } ;

  inline Int convLikeSizeHelper(Int inputShape,
                                Int kernelShape,
                                Int stride,
                                std::array<Int,2> padding,
                                Int dilation = 1)
  {
    auto kernelExtent = dilation*(kernelShape - 1) + 1 ;
    return (inputShape - kernelExtent + padding[0] + padding[1]) / stride + 1 ;
  }

} }

#endif /* nnoperation_hpp */
