// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
Copyright (C) 2015-17 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnconv.hpp"
#include "nnbias.hpp"
#include "impl/dispatcher.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include "impl/im2row.hpp"
#include <cassert>
#include <string>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;
using namespace std ;

template<DeviceType deviceType, DataType dataType> struct ConvolutionForward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionBackward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionTransposeForward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionTransposeBackward ;
template<DataType dataType> struct ConvolutionForwardCudnn ;
template<DataType dataType> struct ConvolutionBackwardCudnn ;

template<DeviceType deviceType, DataType dataType> struct SubsampleForward ;
template<DeviceType deviceType, DataType dataType> struct SubsampleBackward ;

template<DeviceType deviceType, DataType dataType> struct FullyConnectedForward ;
template<DeviceType deviceType, DataType dataType> struct FullyConnectedBackward ;

#include "impl/nnconv_blas.cpp"
#include "impl/nnconv_subsample_blas.cpp"
#include "impl/nnconv_fullyconnected_blas.cpp"

#if ENABLE_GPU
#include "impl/nnconv_subsample_gpu.cpp"
#endif

#if ENABLE_CUDNN
#include "impl/nnconv_cudnn.cpp"
#endif

// -------------------------------------------------------------------
/// MARK: - Convolution
// -------------------------------------------------------------------

Convolution::Convolution(Context &context,
                         Int strideY, Int strideX,
                         Int padTop, Int padBottom,
                         Int padLeft, Int padRight,
                         Int dilateY, Int dilateX)
: ConvolutionLike(context,2)
{
  setStride({strideY, strideX}) ;
  setPadding({padTop, padBottom, padLeft, padRight}) ;
  setDilation({dilateY, dilateX}) ;
}

Convolution::Convolution(Context &context)
: ConvolutionLike(context,2), dilation((size_t)getNumSpatialDimensions(),1)
{ }

vl::ErrorCode
Convolution::setDilation(vector<Int> const& dilation)
{
  // Dilation must be positive.
  if (any_of(begin(dilation),end(dilation),[](Int x){return x <= 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "Convolution: a dilation parameter is not positive.") ;
  }

  // There must one dilation per spatial dimension.
  if (Int(dilation.size()) == getNumSpatialDimensions()) {
    this->dilation = dilation ;
    copy(begin(dilation),end(dilation),begin(this->dilation)) ;
  }
  else if (dilation.size() == 1) {
    fill(begin(this->dilation),end(this->dilation),dilation[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument,
     "DILATION is neither scalar nor has the same cardinality"
     "as the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

vl::ErrorCode
Convolution::forwardShape(TensorShape &output,
                          TensorShape const& input,
                          TensorShape const& filter,
                          TensorShape const& bias) const
{
  output.clear() ;

  // The input tensor cannot be empty.
  if (input.isEmpty()) {
    return getContext().setError
    (VLE_TensorShapeMismatch, "Convolution: the INPUT tensor is empty.") ;
  }

  // We pretend all tensors have an infinite number of dimensions,
  // potentially singleton.
  Int ns = getNumSpatialDimensions() ;

  // The tensors should have ns+2 dimensions. Todo: we may relax that by implicitly
  // folding the excess dimensions.
  if (input.getNumDimensions() > ns + 2) {
    return getContext().setError(VLE_TensorShapeMismatch,
                                 "Convolution: INPUT has too many dimensions.") ;
  }
  if (filter.getNumDimensions() > ns + 2) {
    return getContext().setError(VLE_TensorShapeMismatch,
                                 "Convolution: FILTER has too many dimensions.") ;
  }

  // Check the filters.
  bool hasFilter = !filter.isEmpty() ;
  if (hasFilter) {
    Int inputNumChannels = input.getDimension(ns)  ;
    Int filterNumChannels = filter.getDimension(ns) ;
    Int numFilters = filter.getDimension(ns+1) ;
    Int numGroups = inputNumChannels / filterNumChannels ;
    if (numGroups * filterNumChannels != inputNumChannels) {
      output = TensorShape() ; // set to null
      return getContext().setError
      (VLE_TensorShapeMismatch,
       "Convolution: the number of channels of FILTER does not divide the one of INPUT.") ;
    }
    Int numFiltersPerGroup = numFilters / numGroups ;
    if (numFiltersPerGroup * numGroups != numFilters) {
      return  getContext().setError
      (VLE_TensorShapeMismatch,
       "Convolution: the number of filters in the bank is not divisible by the number of filter groups.") ;
    }
    for (Int d = 0 ; d < ns ; ++d) {
      Int odim = convLikeSizeHelper(input.getDimension(d),
                                    filter.getDimension(d),
                                    getStride(d),
                                    {getPadding(2*d),getPadding(2*d+1)},
                                    getDilation(d)) ;
      if (odim <= 0) {
        output.clear() ;
        return  getContext().setError
        (VLE_TensorShapeMismatch,
         "Convolution: the spatial dimensions of INPUT are too small for FILTER and the convolution parameters.") ;
      }
      if (filter.getDimension(d) <= getPadding(2*d) ||
          filter.getDimension(d) <= getPadding(2*d+1)) {
        output.clear() ;
        return  getContext().setError
        (VLE_TensorShapeMismatch,
         "Convolution: one of FILTER dimensions is not larger than the corresponding PADDING.") ;
      }
      output.setDimension(d,odim) ;
    }
    output.setDimension(ns, numFilters) ;
    output.setDimension(ns+1, input.getDimension(ns+1)) ;
  } else {
    // Bias / subsample mode.
    for (Int d = 0 ; d < ns ; ++d) {
      Int odim = convLikeSizeHelper(input.getDimension(d),
                                    1,
                                    getStride(d),
                                    {getPadding(2*d),getPadding(2*d+1)},
                                    1) ;
      if (odim <= 0) {
        output.clear() ;
        return  getContext().setError
        (VLE_TensorShapeMismatch,
         "Convolution: the spatial dimensions of INPUT are too small for the convolution parameters.") ;
      }
      output.setDimension(d,odim) ;
    }
    output.setDimension(ns, input.getDimension(ns)) ;
    output.setDimension(ns+1, input.getDimension(ns+1)) ;
  }

  // Check the bias.
  bool hasBias = !bias.isEmpty() ;
  if (hasBias) {
    if (bias.getNumElements() != output.getNumChannels()) {
      output.clear() ;
      return  getContext().setError
      (VLE_TensorShapeMismatch,
       "Convolution: BIAS does not have a number of elements equal to the number of output feature channels.") ;
    }
  }

  return VLE_Success ;
}

vl::ErrorCode
Convolution::forward(Tensor &output, double outputMult,
                     Tensor const& input, double inputMult,
                     Tensor const& filter,
                     Tensor const& bias) const
{
  ErrorCode error ;

  // Validate arguments.
  if (!check_tensor_compatibility(output,input,filter,bias)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionForward: the tensors have mismatching data or device type.") ;
  }
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape, input, filter, bias)) != VLE_Success) {
    return error ;
  }
  if (output != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "ConvolutionForward: OUTPUT does not have the appropriate dimensions.") ;
  }
  if (!input.isEmpty() && input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionForward: INPUT is not an empty tensor, but it has no data either.") ;
  }
  if (!output.isEmpty() && output.isNull()) {
    return  getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionForward: OUTPUT is not an empty tensor, but it has no data either.") ;
  }
  if (!filter.isEmpty() && filter.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionForward: FILTER is not an empty tensor, but it has no data either.") ;
  }
  if (!bias.isNull() && bias.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionForward: BIAS is not an empty tensor, but it has no data either.") ;
  }

  VLLOG(*this,1)
  << "Convolution: forward"
  << " stride=" << pretty(getStrides())
  << " padding=" << pretty(getPaddings())
  << " dilation=" << pretty(getDilations()) ;

  VLLOG(*this,1)
  << "Convolution: input=" << pretty(input.getDimensions())
  << " filter=" << pretty(filter.getDimensions())
  << " bias=" << pretty(bias.getDimensions())
  << " output=" << pretty(output.getDimensions()) ;

  auto const isFullyConnected = [&](){
    auto isone = [](Int x){return x==1;} ;
    auto iszero = [](Int x){return x==0;} ;
    auto ns = getNumSpatialDimensions() ;
    auto dims = output.getDimensions() ;
    auto paddings = getPaddings() ;
    auto strides = getStrides() ;
    auto dilations = getDilations() ;
    return (all_of(begin(dims), begin(dims)+ns,isone) &&
            all_of(begin(strides),end(strides),isone) &&
            all_of(begin(paddings),end(paddings),iszero) &&
            all_of(begin(dilations),end(dilations),isone) &&
            filter.getDimension(ns) == input.getDimension(ns)) ;
  } ;

  // Filtering.
  Tensor null ;
  if (filter.isEmpty()) {
    // Subsample mode.
    error = dispatch<SubsampleForward>()(*this,output,input) ;
  }
  else if (isFullyConnected()) {
    // Fully-connected mode.
    error = dispatch<FullyConnectedForward>()(*this,output,outputMult,input,inputMult,filter) ;
  } else {
    // Normal mode.
    error = dispatch_cudnn<
    ConvolutionForward,
    ConvolutionForwardCudnn>()
    (*this,output,outputMult,input,inputMult,filter) ;
  }
  if (error != vl::VLE_Success) { return getContext().passError(error,"ConvolutionForward") ; }

  // Bias.
  if (!bias.isEmpty()) {
    error = vl::nn::Bias(getContext()).forward(output,1.0,Tensor(),0,bias,1.0);
  }

  return getContext().passError(error,"ConvolutionForward") ;
}

/// * `derOutput` must be a non-empty, non-null tensor.
///
/// * `derInput`, `derFiler`, and `derBias` must be as in the forward call, or be
///   emtpy in order to skip the computation of the corresponding derivative.
///
/// * `input` and `filter` should be as in the forward call. As an optimization,
///   if `derInput` is not requested, `filter` can be null (forgotten) and
///   if `derFilter` is not requested, then `input` can be forgotten.
///
vl::ErrorCode
Convolution::backward(Tensor &derInput,
                      Tensor &derFilter,
                      Tensor &derBias,
                      Tensor const &input,
                      Tensor const &filter,
                      Tensor const &derOutput) const
{
  ErrorCode error ;

  // Validate arguments.
  if (!check_tensor_compatibility(derInput,derFilter,derBias,input,filter,derOutput)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionBackward: the tensors have mismatching data or device type.") ;
  }

  // Check that we have the output derivative.
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape, input, filter, TensorShape())) != VLE_Success) {
    return error ;
  }
  if (derOutput != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "ConvolutionBackward: DEROUTPUT does not have the appropriate dimensions.") ;
  }
  if (derOutput.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionBackward: DEROUTPUT is null.") ;
  }

  // If the input derivatives are requested, check that we have what we need.
  if (!derInput.isEmpty()) {
    if (derInput.isNull()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERINPUT requested, but the output tensor is null.") ;
    }
    if (static_cast<TensorShape>(derInput) != static_cast<TensorShape>(input)) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERINPUT requested, but its size is not the same as INPUT.") ;
    }
    if (!filter.isEmpty() && filter.isNull()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERINPUT requested, but FILTER is a non-empty null tensor.") ;
    }
  }

  // If the filter derivatives are requested, check that we have what we need.
  if (!derFilter.isEmpty()) {
    if (derFilter.isNull()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERFILTER requested, but the output tensor is null.") ;
    }
    if (static_cast<TensorShape>(derFilter) != static_cast<TensorShape>(derFilter)) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERFILTER requested, but its size is not the same as FILTER.") ;
    }
    if (input.isEmpty() || input.isNull()) {
      return getContext().setError(VLE_IllegalArgument,
                   "ConvolutionBackward: DERFILTER requested, but INPUT is missing.") ;
    }
  }

  VLLOG(*this,1)
  << "Convolution: backward"
  << " stride=" << pretty(getStrides())
  << " padding=" << pretty(getPaddings())
  << " dilation=" << pretty(getDilations()) ;

  VLLOG(*this,1)
  << "Convolution: input=" << pretty(input.getDimensions())
  << " filter=" << pretty(filter.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  VLLOG(*this,1)
  << "Convolution: derInput=" << pretty(derInput.getDimensions())
  << " derFilter=" << pretty(derFilter.getDimensions())
  << " derBias=" << pretty(derBias.getDimensions()) ;

  auto const isFullyConnected = [&](){
    auto isone = [](Int x){return x==1;} ;
    auto iszero = [](Int x){return x==0;} ;
    auto ns = getNumSpatialDimensions() ;
    auto dims = derOutput.getDimensions() ;
    auto paddings = getPaddings() ;
    auto strides = getStrides() ;
    auto dilations = getDilations() ;
    return (all_of(begin(dims), begin(dims)+ns,isone) &&
            all_of(begin(strides),end(strides),isone) &&
            all_of(begin(paddings),end(paddings),iszero) &&
            all_of(begin(dilations),end(dilations),isone) &&
            filter.getDimension(ns) == input.getDimension(ns)) ;
  } ;

  // Filtering.
  Tensor null ;
  if (filter.isEmpty()) {
    // Subsample mode.
    error = dispatch<SubsampleBackward>()(*this,derInput,derOutput) ;
  }
  else if (isFullyConnected()) {
    // Fully-connected mode.
    error = dispatch<FullyConnectedBackward>()(*this,derInput,derFilter,input,filter,derOutput) ;
  }
  else {
    // Normal mode.
    error = dispatch_cudnn<
    ConvolutionBackward,
    ConvolutionBackwardCudnn>()
    (*this,derInput,derFilter,input,filter,derOutput) ;
  }

  // If the bias derivaties are requested, check that we have what we need.
  if (!derBias.isEmpty()) {
    if (derBias.isNull()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERBIAS requested, but the output tensor is null.") ;
    }
    if (derBias.getNumElements() != derOutput.getNumChannels()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERBIAS requested, but the output tensro has an incorrect size.") ;
    }
    error = vl::nn::Bias(getContext()).backward(null,0,derBias,0,0,1,derOutput) ;
  }
  return getContext().passError(error,"ConvolutionBackward") ;
}

/// MARK: - Convolution Transpose

/// Todo: make obsolete
ConvolutionTranspose::ConvolutionTranspose(Context &context,
                                           Int upsampleY,
                                           Int upsampleX,
                                           Int cropTop,
                                           Int cropBottom,
                                           Int cropLeft,
                                           Int cropRight)
:
Operation {context},
numSpatialDimensions {2},
upsampling {upsampleY, upsampleX},
cropping {cropTop, cropBottom, cropLeft, cropRight},
numFilterGroups {1}
{ }

ConvolutionTranspose::ConvolutionTranspose(Context &context)
: Operation(context),
numSpatialDimensions(2),
numFilterGroups(1)
{
  cropping.fill(0) ;
  upsampling.fill(1) ;
}

vl::ErrorCode
ConvolutionTranspose::setUpsampling(vector<Int> const& upsampling)
{
  // Usampling must be positive.
  if (any_of(begin(upsampling),end(upsampling),[](Int x){return x <= 0;})) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionTranspose: an element of UPSAMPLING is not positive.") ;
  }

  // There must one upsampling per spatial dimension.
  if (Int(upsampling.size()) == getNumSpatialDimensions()) {
    copy(begin(upsampling),end(upsampling),begin(this->upsampling)) ;
  }
  else if (upsampling.size() == 1) {
    this->upsampling.fill(upsampling[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "ConvolutionTranspose: UPSAMPLING is neither scalar nor has the same"
     " cardinality as the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

vl::ErrorCode
ConvolutionTranspose::setCropping(vector<Int> const& cropping)
{
  // Cropping must be non-negative.
  if (any_of(begin(cropping),end(cropping),[](Int x){return x < 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "ConvolutionTranspose: An element of CROPPING is less than 0.") ;
  }
  // There must one stride per spatial dimension.
  if (Int(cropping.size()) == 2*numSpatialDimensions) {
    copy(begin(cropping),end(cropping),begin(this->cropping)) ;
  }
  else if (cropping.size() == 1) {
    this->cropping.fill(cropping[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "ConvolutionTranspose: CROPPING is neither scalar nor has the cardinality"
     " of twice the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

vl::ErrorCode
ConvolutionTranspose::setNumFilterGroups(Int numFilterGroups)
{
  // NumFilterGroups must be non-negative.
  if (numFilterGroups < 1) {
    return getContext().setError
    (VLE_IllegalArgument, "ConvolutionTranspose: NUMFILTERGROUPS is less than 1.") ;
  }
  this->numFilterGroups = numFilterGroups ;
  return VLE_Success ;
}

vl::ErrorCode
ConvolutionTranspose::forwardShape(TensorShape &output,
                                   TensorShape const& input,
                                   TensorShape const& filter,
                                   TensorShape const& bias) const
{
  output.clear() ;

  // The input tensor cannot be empty.
  if (input.isEmpty()) {
    return getContext().setError
    (VLE_TensorShapeMismatch, "ConvolutionTranspose: the INPUT tensor is empty.") ;
  }
  if (filter.isEmpty()) {
    return getContext().setError
    (VLE_TensorShapeMismatch, "ConvolutionTranspose: FILTER is empty.") ;
  }

  // We pretend all tensors have an infinite number of dimensions,
  // potentially singleton.
  Int ns = getNumSpatialDimensions() ;

  // The tensors should have ns+2 dimensions. Todo: we may relax that by implicitly
  // folding the excess dimensions.
  if (input.getNumDimensions() > ns + 2) {
    return getContext().setError(VLE_TensorShapeMismatch,
                                 "ConvolutionTranspose: INPUT has too many dimensions.") ;
  }
  if (filter.getNumDimensions() > ns + 2) {
    return getContext().setError(VLE_TensorShapeMismatch,
                                 "ConvolutionTranspose: FILTER has too many dimensions.") ;
  }

  // Check the filters.
  if (filter.getCardinality() % getNumFilterGroups() != 0) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "The number of filter groups does not divide the number of channels in FILTERS.") ;
  }
  if (filter.getCardinality() != input.getNumChannels()) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "The total number of channels in FILTERS is not the same as the number of channels of INPUT.") ;
  }

  Int inputNumChannels = input.getDimension(ns)  ;
  Int filterNumChannels = filter.getDimension(ns) ;
  Int numFilters = filter.getDimension(ns+1) ;

  for (Int d = 0 ; d < ns ; ++d) {
    Int odim =
    (input.getDimension(d) - 1) * getUpsampling(d) -
    (getCropping(2*d) + getCropping(2*d+1)) + filter.getDimension(d) ;
    if (odim <= 0) {
      output.clear() ;
      return getContext().setError
      (VLE_TensorShapeMismatch,
       "ConvolutionTranspose: the spatial dimensions of INPUT are"
       " too small for FILTER and the convolution parameters.") ;
    }
    output.setDimension(d,odim) ;
  }
  output.setDimension(ns, filter.getNumChannels() * getNumFilterGroups()) ;
  output.setDimension(ns+1, input.getDimension(ns+1)) ;

  // Check the bias.
  bool hasBias = !bias.isEmpty() ;
  if (hasBias) {
    if (bias.getNumElements() != output.getNumChannels()) {
      output.clear() ;
      return getContext().setError
      (VLE_TensorShapeMismatch,
       "ConvolutionTranspose: BIAS does not have a number of elements"
       " equal to the number of output feature channels.") ;
    }
  }

  return VLE_Success ;
}

vl::ErrorCode
ConvolutionTranspose::forward(Tensor &output,
                              Tensor const& input,
                              Tensor const& filter,
                              Tensor const& bias) const
{
  VLLOG(*this,1)
  << "ConvolutionTranspose: forward"
  << " upsampling=" << pretty(getUpsamplings())
  << " cropping=" << pretty(getCroppings())
  << " numFilterGroups=" << getNumFilterGroups() ;

  VLLOG(*this,1)
  << "ConvolutionTranspose: input=" << pretty(input.getDimensions())
  << " filter=" << pretty(filter.getDimensions())
  << " bias=" << pretty(bias.getDimensions())
  << " output=" << pretty(output.getDimensions()) ;

  return getContext().passError
  (dispatch<ConvolutionTransposeForward>()
   (*this,output,input,filter,bias),
   "ConvolutionTransposeForward") ;
}

vl::ErrorCode
ConvolutionTranspose::backward(Tensor &derInput,
                               Tensor &derFilter,
                               Tensor &derBias,
                               Tensor const &input,
                               Tensor const &filter,
                               Tensor const &derOutput) const
{
  VLLOG(*this,1)
  << "ConvolutionTranspose: backward"
  << " upsampling=" << pretty(getUpsamplings())
  << " cropping=" << pretty(getCroppings())
  << " numFilterGroups=" << getNumFilterGroups() ;

  VLLOG(*this,1)
  << "ConvolutionTranspose: input=" << pretty(input.getDimensions())
  << " filter=" << pretty(filter.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  VLLOG(*this,1)
  << "ConvolutionTranspose: derInput=" << pretty(derInput.getDimensions())
  << " derFilter=" << pretty(derFilter.getDimensions())
  << " derBias=" << pretty(derBias.getDimensions()) ;

  return getContext().passError
  (dispatch<ConvolutionTransposeBackward>()
  (*this,derInput,derFilter,derBias,input,filter,derOutput),
   "ConvolutionTransposeBackward") ;
}


