//
//  nnoperation.cpp
//  mcn-lib-cpu
//
//  Created by Andrea Vedaldi on 16/10/2017.
//  Copyright © 2017 Andrea Vedaldi. All rights reserved.
//

#include "nnoperation.hpp"
#include <algorithm>

using namespace vl ;
using namespace vl::nn ;
using namespace std ;

ConvolutionLike::ConvolutionLike(Context& context, Int numSpatialDimensions)
: Operation(context), numSpatialDimensions(numSpatialDimensions)
{
  stride.fill(1) ;
  padding.fill(0) ;
}

vl::ErrorCode ConvolutionLike::setStride(vector<Int> const& stride)
{
  // Stride must be positive.
  if (any_of(begin(stride),end(stride),[](Int x){return x <= 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "An element of STRIDE is less than 1.") ;
  }
  // There must one stride per spatial dimension.
  if (Int(stride.size()) == numSpatialDimensions) {
    copy(begin(stride),end(stride),begin(this->stride)) ;
  }
  else if (stride.size() == 1) {
    this->stride.fill(stride[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "STRIDE is neither scalar nor has the same cardinality as the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

vl::ErrorCode ConvolutionLike::setPadding(vector<Int> const& padding)
{
  // Padding must be non-negative.
  if (any_of(begin(padding),end(padding),[](Int x){return x < 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "An element of PADDING is less than 0.") ;
  }
  // There must one stride per spatial dimension.
  if (Int(padding.size()) == 2*numSpatialDimensions) {
    copy(begin(padding),end(padding),begin(this->stride)) ;
  }
  else if (padding.size() == 1) {
    this->padding.fill(stride[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "PADDING is neither scalar nor has the cardinality of twice the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

