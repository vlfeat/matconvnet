// @file nnbnorm.cu
// @brief Batch normalization block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Sebastien Ehrhardt and Andrea Vedaldi.
Copyright (C) 2017 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbnorm.hpp"
#include "impl/dispatcher.hpp"
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct BatchNormForward ;
template<DeviceType deviceType, DataType dataType> struct BatchNormForwardWithMoment ;
template<DeviceType deviceType, DataType dataType> struct BatchNormBackward ;
template<DeviceType deviceType, DataType dataType> struct BatchNormBackwardWithMoment ;

template<DataType dataType> struct BatchNormForwardCudnn ;
template<DataType dataType> struct BatchNormForwardWithMomentCudnn ;
template<DataType dataType> struct BatchNormBackwardCudnn ;
template<DataType dataType> struct BatchNormBackwardWithMomentCudnn ;

#if ENABLE_GPU
#include "nnbnorm_gpu.cu"
#endif

#if ENABLE_CUDNN
#include "nnbnorm_cudnn.cu"
#endif

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

// Compute moments (means and sigmas) from the batch data
// WH is the product of the data width and height
// moments is a 2 x numChannels array with means and sigmas

template<typename T> inline void
compute_moment(T * moments,
               T const * data,
               Int WH,
               Int numChannels,
               Int cardinality,
               T epsilon)
{
  memset(moments, 0, sizeof(T) * 2 * as_unsigned(numChannels)) ;
  Int mass = WH * cardinality ;
  for(Int channel = 0; channel < numChannels; ++channel) {
    for(Int element = 0; element < cardinality; ++element) {
      for(Int wh = 0; wh < WH; ++wh){
        T x = data[wh + channel*WH + element*(numChannels*WH)] ;
        moments[channel] += x ; // mean
        moments[channel + numChannels] += x * x; // sigma
      }
    }
  }
  for(Int i = 0; i < numChannels; ++i) {
    T mean = moments[i] / mass ;
    T sigma2 = std::max((T).0, moments[i + numChannels]/mass - mean*mean) ;
    moments[i] = mean ;
    moments[i + numChannels] = std::sqrt(sigma2 + (T)epsilon);
  }
}

// This version assumes that the moment tensor is precomputed.
template<typename T> inline void
compute_ders(T * derMultipliers,
             T * derBiases,
             T const * moments,
             T const * data,
             T const * derOutput,
             Int WH, Int numChannels, Int cardinality,
             T epsilon)
{
  memset(derMultipliers, 0, sizeof(T) * (size_t)numChannels) ;
  memset(derBiases, 0, sizeof(T) * (size_t)numChannels) ;
  for(Int channel = 0; channel < numChannels; ++channel){
    for(Int element = 0; element < cardinality; ++element ){
      for(Int wh = 0; wh < WH; ++wh){
        auto offset = wh + channel * WH + element * (WH*numChannels) ;
        derMultipliers[channel] += derOutput[offset] * data[offset];
        derBiases[channel] += derOutput[offset];
      }
    }
  }
  for(Int i = 0; i < numChannels; ++i) {
    T mean = moments[i] ;
    T sigma = moments[i + numChannels] ;
    derMultipliers[i] = (derMultipliers[i] - mean*derBiases[i]) / sigma;
  }
}

template<typename T> inline void
compute_ders_and_moments(T * derMultipliers,
                         T * derBiases,
                         T * moments,
                         T const * data,
                         T const * derOutput,
                         Int WH,
                         Int numChannels,
                         Int cardinality,
                         T epsilon)
{
  memset(derMultipliers, 0, sizeof(T) * (size_t)numChannels) ;
  memset(derBiases, 0, sizeof(T) * (size_t)numChannels) ;
  memset(moments, 0, sizeof(T) * 2 * (size_t)numChannels) ;
  for(Int channel = 0; channel < numChannels; ++channel) {
    for(Int element = 0; element < cardinality; ++element) {
      for(Int wh = 0; wh < WH; ++wh){
        auto offset = wh + channel * WH + element * (WH*numChannels) ;
        moments[channel] += data[offset] ;
        moments[channel + numChannels] += data[offset] * data[offset];
        derMultipliers[channel] += derOutput[offset] * data[offset];
        derBiases[channel] += derOutput[offset];
      }
    }
  }

  T mass = (T)(WH*cardinality) ;
  for(Int i = 0; i < numChannels; ++i) {
    T mean = moments[i] / mass ;
    T sigma2 = std::max((T).0, moments[i + numChannels]/mass - mean*mean) ;
    T sigma = std::sqrt(sigma2 + (T)epsilon);
    moments[i] = mean ;
    moments[i + numChannels] = sigma ;
    derMultipliers[i] = (derMultipliers[i] - mean*derBiases[i]) / sigma;
  }
}

template<typename T> inline void
batch_normalize_backward(T * derData,
                         T const * moments,
                         T const * data,
                         T const * multipliers,
                         T const * derMultipliers,
                         T const * derBiases,
                         T const * derOutput,
                         Int WH,
                         Int numChannels,
                         Int cardinality)
{
  T mass = (T)(WH*cardinality) ;
  for (Int channel = 0; channel < numChannels; ++channel) {
    T mean = moments[channel] ;
    T sigma = moments[channel + numChannels] ;

    T muz = derBiases[channel]/mass ;
    T G1 = multipliers[channel]/sigma ;
    T G2 = G1 * derMultipliers[channel]/(mass*sigma) ;

    for (Int element = 0; element < cardinality; ++element){
      for (Int wh = 0; wh < WH; ++wh){
        auto offset = wh + channel * WH + element * (WH*numChannels) ;
        derData[offset] = G1 * (derOutput[offset] - muz) - G2 * (data[offset]-mean) ;
      }
    }
  }
}

// -------------------------------------------------------------------
/// MARK: - Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BatchNormForwardWithMoment<VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm const &op,
                           Tensor &output,
                           Tensor const &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias)
  {
    static const std::string signature = std::string("BatchNormForwardWithMoments[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int numChannels = input.getNumChannels() ;
    Int cardinality = input.getCardinality() ;
    Int WH = height * width ;

    auto outputData = (type*)output.getMemory() ;
    auto momentData = (type const*)moment.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;
    auto biasData = (type const*)bias.getMemory() ;

    assert(outputData) ;
    assert(momentData) ;
    assert(inputData) ;
    assert(multiplierData) ;
    assert(biasData) ;

    for(Int channel = 0; channel < numChannels; ++channel) {
      type mean = momentData[channel] ;
      type sigma = momentData[channel + numChannels] ;
      type bias = biasData[channel];
      type coefficient = multiplierData[channel] / sigma ;

      for(decltype(cardinality) element = 0; element < cardinality; ++element) {
        for(decltype(WH) wh = 0; wh < WH; ++wh){
          auto offset = wh + channel * WH + element * numChannels * WH ;
          outputData[offset] = coefficient * (inputData[offset] - mean) + bias ;
        }
      }
    }
    return VLE_Success ;
  }
} ;

template<DataType dataType>
struct BatchNormForward<VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm const &op,
                           Tensor &output,
                           Tensor &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias)
  {
    static const std::string signature = std::string("BatchNormForward[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int numChannels = input.getNumChannels() ;
    Int cardinality = input.getCardinality() ;
    auto inputData = (type const*)input.getMemory() ;

    assert(inputData) ;

    // Allocate memory for the moments if needed.
    Tensor ownMoment(moment) ;
    if (ownMoment.isNull()) {
      auto * buffer = (type*)op.getContext().getWorkspace
      (vl::VLDT_CPU, sizeof(type)*2*size_t(numChannels)) ;
      if (!buffer) {
        return op.getContext().setError
        (VLE_OutOfMemory, "BatchNormForward: could not allocate enough memory.") ;
      }
      ownMoment.setMemory(buffer) ;
    }

    // Compute the moments.
    auto momentData = (type*)ownMoment.getMemory() ;
    compute_moment<type>(momentData, inputData,
                         width*height, numChannels, cardinality,
                         (type)op.getEpsilon()) ;

    // Compute output.
    return op.getContext().passError
    (BatchNormForwardWithMoment<vl::VLDT_CPU,dataType>()
     (op,output,ownMoment,input,multiplier,bias),
     signature.c_str()) ;
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct BatchNormBackwardWithMoment<VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm const &op,
                           Tensor &derInput,
                           Tensor &derMultiplier,
                           Tensor &derBias,
                           Tensor const &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias,
                           Tensor const &derOutput)
  {
    static const std::string signature = std::string("BatchNormBackwardWithMoment[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int numChannels = input.getNumChannels() ;
    Int cardinality = input.getCardinality() ;
    Int WH = height * width ;

    auto derInputData = (type*)derInput.getMemory() ;
    auto derMultiplierData = (type*)derMultiplier.getMemory() ;
    auto derBiasData = (type*)derBias.getMemory() ;
    auto momentData = (type const*)moment.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

    assert(derInputData) ;
    assert(derMultiplierData) ;
    assert(derBiasData) ;
    assert(momentData) ;
    assert(inputData) ;
    assert(multiplierData) ;
    assert(derOutputData) ;

    // Compute derMultipliers, derBiases, muz, and moments.
    compute_ders<type>(derMultiplierData, derBiasData,
                       momentData, inputData, derOutputData,
                       WH, numChannels, cardinality,
                       (type)op.getEpsilon());

    // Compute derData.
    batch_normalize_backward<type>(derInputData,
                                   momentData, inputData,
                                   multiplierData,
                                   derMultiplierData, derBiasData, derOutputData,
                                   WH, numChannels, cardinality);
    return VLE_Success ;
  }
} ;

template<DataType dataType>
struct BatchNormBackward<VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm const &op,
                           Tensor &derInput,
                           Tensor &derMultiplier,
                           Tensor &derBias,
                           Tensor &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias,
                           Tensor const &derOutput)
  {
    static const std::string signature = std::string("BatchNormBackward[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int numChannels = input.getNumChannels() ;
    Int cardinality = input.getCardinality() ;
    Int WH = height * width ;

    auto derInputData = (type*)derInput.getMemory() ;
    auto derMultiplierData = (type*)derMultiplier.getMemory() ;
    auto derBiasData = (type*)derBias.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

     // Allocate memory for the moments if needed.
    Tensor ownMoment(moment) ;
    if (ownMoment.getMemory() == NULL) {
      auto buffer = (type*)op.getContext().getWorkspace
      (vl::VLDT_CPU, sizeof(type)*2*size_t(numChannels)) ;
      if (!buffer) {
        return op.getContext().setError
        (VLE_OutOfMemory, "BatchNormBackward: could not allocate enough memory.") ;
      }
      ownMoment.setMemory(buffer) ;
    }
    auto momentData = (type*)ownMoment.getMemory() ;

    // Compute derMultipliers, derBiases, and moments.
    compute_ders_and_moments<type>(derMultiplierData, derBiasData, momentData,
                                   inputData, derOutputData,
                                   WH, numChannels, cardinality,
                                   (type)op.getEpsilon());

    // Compute derData.
    batch_normalize_backward<type>(derInputData,
                                   momentData, inputData,
                                   multiplierData,
                                   derMultiplierData, derBiasData, derOutputData,
                                   WH, numChannels, cardinality);
    return VLE_Success ;
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Driver
// -------------------------------------------------------------------

BatchNorm::BatchNorm(Context &context,
                     double epsilon)
: Operation(context), epsilon(epsilon)
{ }

BatchNorm::BatchNorm(Context &context)
: Operation(context), epsilon(1e-4)
{ }

vl::ErrorCode
BatchNorm::setEpsilon(double epsilon) {
  if (epsilon < 0) {
    return getContext().setError(VLE_IllegalArgument, "EPSILON is negative.") ;
  }
  this->epsilon = epsilon ;
  return VLE_Success ;
}

vl::ErrorCode
BatchNorm::forwardShape(vl::TensorShape &output,
                        vl::TensorShape &moments,
                        vl::TensorShape const &input) const
{
  output.clear() ;
  moments.clear() ;
  Int const ns = 2 ;
  if (input.getNumDimensions() > ns + 2) {
    return getContext().setError(VLE_TensorShapeMismatch,
                                 "INPUT has too many dimensions.") ;
  }
  output = input ;
  moments = {input.getNumChannels(), 2} ;
  return VLE_Success ;
}

template <bool optionalMoments>
static vl::ErrorCode
check_helper(BatchNorm const& op,
             Tensor const &output,
             Tensor const &moment,
             Tensor const &input,
             Tensor const &multiplier,
             Tensor const &bias)
{
  // Check the tensor consistency.
  if (!check_tensor_compatibility(output,moment,input,multiplier,bias)) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "The tensors have mismatching data or device type.") ;
  }

  // Check the data.
  if (output.isEmpty() | output.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "OUTPUT or DEROUTPUT is empty or null.") ;
  }
  if (optionalMoments) {
    if (!moment.isEmpty() && moment.isNull()) {
      return op.getContext().setError
      (VLE_IllegalArgument,
       "MOMENT is non empty but null.") ;
    }
  } else {
    if (moment.isEmpty() || moment.isNull()) {
      return op.getContext().setError
      (VLE_IllegalArgument,
      "MOMENT is empty or null.") ;
    }
  }
  if (input.isEmpty() | input.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "INPUT is emtpy or null.") ;
  }
  if (multiplier.isEmpty() | multiplier.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "MULTIPLIER is emtpy or null.") ;
  }
  if (bias.isEmpty() | bias.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "BIAS is emtpy or null.") ;
  }

  // Check the tensor shape.
  vl::ErrorCode error ;
  TensorShape outputShape ;
  TensorShape momentShape ;
  if ((error = op.forwardShape(outputShape, momentShape, input.getShape())) != VLE_Success) {
    return error ;
  }
  if (output.getShape() != outputShape) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "OUTPUT or DEROUTPUT do not have the appropriate dimensions.") ;
  }
  if (!moment.isEmpty() && (moment.getNumElements() != momentShape.getNumElements())) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "MOMENT does not have the appropriate dimensions.") ;
  }
  if (bias.getNumElements() != input.getNumChannels()) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "BIAS does not have the appropriate dimensions.") ;
  }
  if (multiplier.getNumElements() != input.getNumChannels()) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "MULTIPLIER does not have the appropriate dimensions.") ;
  }
  return VLE_Success ;
}

template <bool optionalMoments>
static vl::ErrorCode
check_helper_backward(BatchNorm const& op,
                      Tensor const &derInput,
                      Tensor const &derMultiplier,
                      Tensor const &derBias,
                      Tensor const &moment,
                      Tensor const &input,
                      Tensor const &multiplier,
                      Tensor const &bias,
                      Tensor const &derOutput)
{
  vl::ErrorCode error = check_helper<optionalMoments>(op,derOutput,moment,input,multiplier,bias) ;
  if (error != vl::VLE_Success) {
    return error ;
  }
  // Check the tensor consistency.
  if (!check_tensor_compatibility(derInput,derMultiplier,derBias,derOutput)) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "The tensors have mismatching data or device type.") ;
  }

  // Check the data.
  if (derInput.isEmpty() | derInput.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "DERINPUT is empty or null.") ;
  }
  if (derMultiplier.isEmpty() | derMultiplier.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "DERMULTIPLIER is emtpy or null.") ;
  }
  if (derBias.isEmpty() | derBias.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "DERBIAS is emtpy or null.") ;
  }

  // Check the tensor shape.
  if (derInput.getShape() != input.getShape()) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "DERINPUT does not have the appropriate dimensions.") ;
  }
  if (derBias.getNumElements() != input.getNumChannels()) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "DERBIAS does not have the appropriate dimensions.") ;
  }
  if (derMultiplier.getNumElements() != input.getNumChannels()) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "DERMULTIPLIER does not have the appropriate dimensions.") ;
  }
  return VLE_Success ;
}

vl::ErrorCode
BatchNorm::forward(Tensor &output,
                   Tensor &moment,
                   Tensor const &input,
                   Tensor const &multiplier,
                   Tensor const &bias) const
{
  vl::ErrorCode error = check_helper<true>(*this,output,moment,input,multiplier,bias) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"BatchNormForward") ;
  }

  VLLOG(*this,1)
  << "BatchNormForward: forward"
  << " epsilon=" << getEpsilon()
  << " moment=" << pretty(moment.getDimensions())
  << " input=" << pretty(input.getDimensions()) ;

  VLLOG(*this,1)
  << "BatchNormForward:"
  << " multiplier=" << pretty(multiplier.getDimensions())
  << " bias=" << pretty(bias.getDimensions()) ;

  return getContext().passError
  (dispatch_cudnn<
   BatchNormForward,
   BatchNormForwardCudnn>()
   (*this,output,moment,input,multiplier,bias),
   "BatchNormForward") ;
}

vl::ErrorCode
BatchNorm::forwardWithMoment(Tensor &output,
                             Tensor const &moment,
                             Tensor const &input,
                             Tensor const &multiplier,
                             Tensor const &bias) const
{
  vl::ErrorCode error = check_helper<false>(*this,output,moment,input,multiplier,bias) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"BatchNormForwardWithMoments") ;
  }

  VLLOG(*this,1)
  << "BatchNormForwardWithMoment:"
  << " epsilon=" << getEpsilon()
  << " moment=" << pretty(moment.getDimensions())
  << " input=" << pretty(input.getDimensions()) ;

  VLLOG(*this,1)
  << "BatchNormForwardWithMoment:"
  << " multiplier=" << pretty(multiplier.getDimensions())
  << " bias=" << pretty(bias.getDimensions()) ;

  return getContext().passError
  (dispatch_cudnn<
  BatchNormForwardWithMoment,
  BatchNormForwardWithMomentCudnn>()
  (*this,output,moment,input,multiplier,bias),
   "BatchNormForwardWithMoment") ;
}


vl::ErrorCode
BatchNorm::backward(Tensor &derInput,
                    Tensor &derMultiplier,
                    Tensor &derBias,
                    Tensor &moment,
                    Tensor const &input,
                    Tensor const &multiplier,
                    Tensor const &bias,
                    Tensor const &derOutput) const
{
  vl::ErrorCode error = check_helper_backward<true>
  (*this,derInput,derMultiplier,derBias,moment,input,multiplier,bias,derOutput) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"BatchNormBackward") ;
  }

  VLLOG(*this,1)
  << "BatchNormBackward:"
  << " epsilon=" << getEpsilon()
  << " derInput=" << pretty(derInput.getDimensions())
  << " derMultiplier=" << pretty(derMultiplier.getDimensions())
  << " derBias=" << pretty(derBias.getDimensions())
  << " moment=" << pretty(derBias.getDimensions()) ;

  VLLOG(*this,1)
  << "BatchNormBackward:"
  << " input=" << pretty(input.getDimensions())
  << " multiplier=" << pretty(multiplier.getDimensions())
  << " bias=" << pretty(bias.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  return getContext().passError
  (dispatch_cudnn<
   BatchNormBackward,
   BatchNormBackwardCudnn>()
   (*this,derInput,derMultiplier,derBias,moment,input,multiplier,bias,derOutput),
   "BatchNormBackward") ;
}

vl::ErrorCode
BatchNorm::backwardWithMoment(Tensor &derInput,
                              Tensor &derMultiplier,
                              Tensor &derBias,
                              Tensor const &moment,
                              Tensor const &input,
                              Tensor const &multiplier,
                              Tensor const &bias,
                              Tensor const &derOutput) const
{
  vl::ErrorCode error = check_helper_backward<false>
  (*this,derInput,derMultiplier,derBias,moment,input,multiplier,bias,derOutput) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"BatchNormBackwardWithMoments") ;
  }

  VLLOG(*this,1)
  << "BatchNormBackwardWithMoments:"
  << " epsilon=" << getEpsilon()
  << " derInput=" << pretty(derInput.getDimensions())
  << " derMultiplier=" << pretty(derMultiplier.getDimensions())
  << " derBias=" << pretty(derBias.getDimensions())
  << " moment=" << pretty(derBias.getDimensions()) ;

  VLLOG(*this,1)
  << "BatchNormBackwardWithMoments:"
  << " input=" << pretty(input.getDimensions())
  << " multiplier=" << pretty(multiplier.getDimensions())
  << " bias=" << pretty(bias.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  return getContext().passError
  (dispatch_cudnn<
  BatchNormBackwardWithMoment,
  BatchNormBackwardWithMomentCudnn>()
  (*this,derInput,derMultiplier,derBias,moment,input,multiplier,bias,derOutput),
   "BatchNormBackwardWithMoments") ;
}
