// @file nnfullyconnected.cu
// @brief Fully-connected block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnfullyconnected.hpp"
#include "impl/dispatcher.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include <cassert>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct FullyConnectedForward ;
template<DeviceType deviceType, DataType dataType> struct FullyConnectedBackward ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct FullyConnectedForward
{
  vl::ErrorCode operator()
  (FullyConnected const &op,
   Tensor &output,
   Tensor const& input,
   Tensor const& filter,
   Tensor const& bias)
  {
    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    type alpha = 1 ;
    type beta = 0 ;

    if (filter) {
      auto filterVolume = filter.getHeight() * filter.getWidth() * filter.getDepth() ;
      if (input.getSize() == 1) {
        /* one image in the stack */
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.getContext(),
         't',
         as_signed(filterVolume), as_signed(filter.getSize()),
         alpha,
         (type const*)filter.getMemory(), as_signed(filterVolume),
         (type const*)input.getMemory(), 1,
         beta,
         (type*)output.getMemory(), 1) ;
        if (error != vl::VLE_Success) { goto done ; }
      } else {
        /* multiple images in the stack */
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         't', 'n',
         as_signed(filter.getSize()),
         as_signed(input.getSize()),
         as_signed(filterVolume),
         alpha,
         (type const*)filter.getMemory(), as_signed(filterVolume),
         (type const*)input.getMemory(), as_signed(filterVolume),
         beta,
         (type*)output.getMemory(), as_signed(filter.getSize())) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    } else {
      error = vl::impl::operations<deviceType,type>::copy
      ((type*)output.getMemory(),
       (type const*)input.getMemory(),
       input.getNumElements()) ;
    }

    if (bias) {
      type beta = 1 ;
      type const* allOnesMemory = (type*)
      op.getContext().getAllOnes(deviceType,
                            dataType,
                            input.getSize()) ;
      if (allOnesMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }
      error = vl::impl::blas<deviceType,dataType>::gemm
      (op.getContext(), 'n', 'n',
       as_signed(bias.getNumElements()), as_signed(input.getSize()), 1,
       alpha,
       (type*)bias.getMemory(), as_signed(bias.getNumElements()),
       allOnesMemory, 1,
       beta,
       (type*)output.getMemory(), as_signed(bias.getNumElements())) ;
      if (error != vl::VLE_Success) { goto done ; }
    }
  done:
    return op.getContext().passError(error, __func__) ;
  }
};

// -------------------------------------------------------------------
//                                                           Backward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct FullyConnectedBackward
{
  vl::ErrorCode operator()
  (FullyConnected const &op,
   vl::Tensor &derInput,
   vl::Tensor &derFilter,
   vl::Tensor &derBias,
   vl::Tensor const &input,
   vl::Tensor const &filter,
   vl::Tensor const &derOutput)
  {
    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    type alpha = 1 ;
    type beta = 0 ;

    if (filter) {
      auto filterVolume = filter.getHeight() * filter.getWidth() * filter.getDepth() ;

      if (derFilter) {
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         'n', 't',
         as_signed(filterVolume),
         as_signed(filter.getSize()),
         as_signed(input.getSize()),
         alpha,
         (type*)input.getMemory(), as_signed(filterVolume),
         (type*)derOutput.getMemory(), as_signed(filter.getSize()),
         beta,
         (type*)derFilter.getMemory(), as_signed(filterVolume)) ;
        if (error != vl::VLE_Success) { goto done ; }
      }

      if (derInput) {
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         'n', 'n',
         as_signed(filterVolume),
         as_signed(input.getSize()),
         as_signed(filter.getSize()),
         alpha,
         (type*)filter.getMemory(), as_signed(filterVolume),
         (type*)derOutput.getMemory(), as_signed(filter.getSize()),
         beta,
         (type*)derInput.getMemory(), as_signed(filterVolume)) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    } else {
      vl::impl::operations<deviceType,type>::copy
      ((type*)derInput.getMemory(),
       (type const*)derOutput.getMemory(),
       derOutput.getNumElements()) ;
    }

    if (derBias) {
      auto allOnesMemory = (type const*)
      op.getContext().getAllOnes(deviceType,
                            dataType,
                            derOutput.getSize()) ;
      if (allOnesMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }

      error = vl::impl::blas<deviceType, dataType>::gemm
      (op.getContext(),
       'n', 't',
       1,
       as_signed(derOutput.getDepth()),
       as_signed(derOutput.getSize()),
       alpha,
       (type*)allOnesMemory, 1,
       (type*)derOutput.getMemory(), as_signed(derOutput.getDepth()),
       beta,
       (type*)derBias.getMemory(), 1) ;
      if (error != vl::VLE_Success) { goto done ; }

    }
  done:
    return op.getContext().passError(error, __func__) ;
  }
};

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

FullyConnected::FullyConnected(Context &context)
: Operation(context)
{ }

vl::ErrorCode
FullyConnected::forward(Tensor &output,
                        Tensor const& input,
                        Tensor const& filter,
                        Tensor const& bias)
{
  return dispatch<FullyConnectedForward>()
  (*this,output,input,filter,bias) ;
}

vl::ErrorCode
FullyConnected::backward(Tensor &derInput,
                         Tensor &derFilter,
                         Tensor &derBias,
                         Tensor const &input,
                         Tensor const &filter,
                         Tensor const &derOutput)
{
  return dispatch<FullyConnectedBackward>()
  (*this,derInput,derFilter,derBias,input,filter,derOutput) ;
}

