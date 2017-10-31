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
#include <vector>
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
      assert(0 <= index && index < 2*getNumSpatialDimensions()) ;
      return padding[as_unsigned(index)] ;
    }

    Int getStride(Int index) const {
      assert(0 <= index && index < getNumSpatialDimensions()) ;
      return stride[as_unsigned(index)] ;
    }

    std::vector<Int> const& getStrides() const {
      return stride ;
    }

    std::vector<Int> const& getPaddings() const {
      return padding ;
    }

  private:
    Int numSpatialDimensions ;
    std::vector<Int> stride ;
    std::vector<Int> padding ;
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
