// @file nnbias.cu
// @brief Bias block
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbias.hpp"
#include "impl/dispatcher.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>

using namespace std ;
using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct BiasForward ;
template<DeviceType deviceType, DataType dataType> struct BiasBackward ;
template<DataType dataType> struct BiasForwardCudnn ;
template<DataType dataType> struct BiasBackwardCudnn ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct BiasForward
{
  // output <- outputMult * output + inputMult * input + biasMult * bias
  vl::ErrorCode operator()(Bias & op,
                           Tensor &output, double outputMult,
                           Tensor const &input, double inputMult,
                           Tensor const &bias, double biasMult)
  {
    vl::ErrorCode error = VLE_Success ;
    auto numOutputPixels = output.getHeight() * output.getWidth() ;

    typedef typename DataTypeTraits<dataType>::type type ;

    // Broadcast add biasMult * bias.
    if (bias && biasMult != 0) {
      type const* allOnesMemory = (type*) op.getContext().getAllOnes
      (deviceType, dataType, (size_t)numOutputPixels) ;
      if (allOnesMemory == NULL) {
        error = op.getContext().getLastError() ; goto done ;
      }

      for (Int image = 0 ; image < output.getSize() ; ++image) {
        auto outputOffset =
        (output.getHeight()*output.getWidth()*output.getDepth()) * image ;

        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         'n', 'n',
         numOutputPixels, bias.getNumElements(), 1,
         static_cast<type>(biasMult),
         allOnesMemory, numOutputPixels,
         (type*)bias.getMemory(), 1,
         static_cast<type>(outputMult), // alpha
         (type*)output.getMemory() + outputOffset,
         numOutputPixels) ;
        if (error != VLE_Success) { goto done ; }
      }
    }
    else {
      error = vl::impl::operations<deviceType,type>::fill
      ((type*)output.getMemory(), (size_t)output.getNumElements(), 0) ;
      if (error != VLE_Success) { goto done ; }
    }

    // Add inputMult * input.
    if (input && inputMult != 0) {
      error = vl::impl::blas<deviceType,dataType>::axpy
      (op.getContext(),output.getNumElements(),
       static_cast<type>(inputMult),(type const*)input.getMemory(),1,
       (type*)output.getMemory(),1) ;
      if (error != VLE_Success) { goto done ; }
    }

  done:
    return op.getContext().passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct BiasBackward
{
  vl::ErrorCode operator()(Bias &op,
                           Tensor &derInput, double derInputMult,
                           Tensor &derBias, double derBiasMult,
                           double inputMult, double biasMult,
                           Tensor const &derOutput)
  {
    assert(derOutput) ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;

    // Sratch space.
    type const* allOnesMemory = NULL ;
    allOnesMemory = (type*) op.getContext().getAllOnes(deviceType,
                                                  dataType,
                                                  (size_t)numOutputPixels) ;
    if (allOnesMemory == NULL) {
      error = op.getContext().getLastError() ;
      return VLE_OutOfMemory ;
    }

    // Compute derBias.
    if (derBias) {
      // Sum derOutput along the broadcast dimensions. These
      // are x,y, and image.
      for (Int image = 0 ; image < derOutput.getSize() ; ++image) {
        auto derOutputOffset =
        (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

        error = vl::impl::blas<deviceType,dataType>::
        gemv(op.getContext(),
             't',
             numOutputPixels,
             derOutput.getDepth(),
             static_cast<type>(biasMult), // alpha
             (type*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
             allOnesMemory, 1,
             static_cast<type>((image == 0) ? derBiasMult : 1.0), // beta
             (type*)derBias.getMemory(), 1) ;

        if (error != vl::VLE_Success) { return error ; }
      }
    }

    // Compute derInput.
    if (derInput) {
      // Fill with zeros, scale, or leave unchanged.
      if (derInput == 0.0) {
        error = vl::impl::operations<deviceType,type>::fill
        ((type*)derInput.getMemory(), (size_t)derInput.getNumElements(), 0) ;
      }
      else if (derInputMult != 1.0) {
        error = vl::impl::operations<deviceType,type>::copy
        ((type*)derInput.getMemory(),
         (type*)derInput.getMemory(),
         (size_t)derInput.getNumElements(), static_cast<type>(derInputMult)) ;
      }
      if (error != VLE_Success) { goto done ; }

      // Add.
      error = vl::impl::blas<deviceType,dataType>::
      axpy(op.getContext(),derInput.getNumElements(),
           static_cast<type>(inputMult),(type const*)derOutput.getMemory(),1,
           (type*)derInput.getMemory(),1) ;
      if (error != VLE_Success) { goto done ; }
    }

  done:
    return op.getContext().passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_CUDNN
#include "nnbias_cudnn.cu"
#endif

Bias::Bias(Context &context)
: Operation(context)
{ }

vl::ErrorCode
Bias::forward(vl::Tensor &output, double outputMult,
              vl::Tensor const &input, double inputMult,
              vl::Tensor const &bias, double biasMult)
{
  return dispatch_cudnn<
  BiasForward,
  BiasForwardCudnn>()
  (*this,output,outputMult,input,inputMult,bias,biasMult) ;
}

vl::ErrorCode
Bias::backward(vl::Tensor &derInput, double derInputMult,
               vl::Tensor &derBias, double derBiasMult,
               double inputMult, double biasMult,
               vl::Tensor const &derOutput)
{
  return dispatch_cudnn<
  BiasBackward,
  BiasBackwardCudnn>()
  (*this,derInput,derInputMult,derBias,derBiasMult,inputMult,biasMult,derOutput) ;
}
