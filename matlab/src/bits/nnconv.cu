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

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct ConvolutionForward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionBackward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionTransposeForward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionTransposeBackward ;
template<DataType dataType> struct ConvolutionForwardCudnn ;
template<DataType dataType> struct ConvolutionBackwardCudnn ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------
/*
 One image at a time is processed.

 Filters are (optionally) divided in to groups, one for each group of dimensions.


                 patchVolume                  numFilters
                 +-------------------------+   +-----------------------+

                 filterVolume              numFiltersPerGroup
                 +------------+------------+   +-----------+-----------+      +--------+--------+
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |  filter   |           |      |        |        |
                 |            |            |   |  group 1  |     0     |  =   |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------------------+      |        |        |
 numOutputPixels |   grp. 1   |   grp. 2   |   |           |           |      |        |        |
                 |            |            |   |           |  filter   |      |        |        |
                 |            |            |   |     0     |  group 2  |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------+-----------+      |        |        |
                 |            |            |                                  |        |        |
                 |            |            |   filters                        |        |        |
                 |            |            |                                  |        |        |
                 +------------+------------+                                  +--------+--------+

                 temp                                                         output

 */


template<DeviceType deviceType, DataType dataType>
struct ConvolutionForward
{
  vl::ErrorCode operator()
  (Convolution &op,
   Tensor output, double outputMult,
   Tensor const& input, double inputMult,
   Tensor const& filter,
   Tensor const& bias)
  {

    assert(output) ;
    assert(input) ;
    assert(filter) ;

    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    ptrdiff_t numGroups = input.getDepth() / filter.getDepth() ;
    ptrdiff_t numFiltersPerGroup = filter.getSize() / numGroups ;
    ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
    ptrdiff_t filterVolume = filter.getHeight() * filter.getWidth() * filter.getDepth() ;
    ptrdiff_t tempVolume = numOutputPixels * filterVolume * numGroups ;

    type* tempMemory = (type*) op.context.getWorkspace(deviceType, tempVolume * sizeof(type)) ;
    type const* allOnesMemory = (type*) op.context.getAllOnes(deviceType,
                                                           dataType,
                                                           numOutputPixels) ;
    if (tempMemory == NULL || allOnesMemory == NULL) {
      error = op.context.getLastError() ;
      goto done ;
    }

    for (int image = 0 ; image < input.getSize() ; ++image) {

      ptrdiff_t dataOffset = (input.getHeight()*input.getWidth()*input.getDepth()) * image ;
      ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;

      error = vl::impl::im2row<deviceType,type>::forward
      (op.context,
       tempMemory,
       (type*)input.getMemory() + dataOffset,
       input.getHeight(), input.getWidth(), input.getDepth(),
       filter.getHeight(), filter.getWidth(),
       op.strideY, op.strideX,
       op.padTop, op.padBottom, op.padLeft, op.padRight,
       op.dilateY, op.dilateX) ;
      if (error != vl::VLE_Success) { goto done ; }

      for (int g = 0 ; g < numGroups ; ++ g) {
        ptrdiff_t filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
        ptrdiff_t tempGrpOffset = numOutputPixels * filterVolume * g ;
        ptrdiff_t outputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        type alpha = inputMult ;
        type beta = outputMult ;
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.context,
         'n', 'n',
         numOutputPixels, numFiltersPerGroup, filterVolume,
         alpha,
         tempMemory + tempGrpOffset, numOutputPixels,
         (type*)filter.getMemory() + filterGrpOffset, filterVolume,
         beta,
         (type*)output.getMemory() + outputOffset + outputGrpOffset, numOutputPixels) ;
        if (error != vl::VLE_Success) { goto done ; }
      }

      if (bias) {
        type alpha = 1 ;
        type beta = 1 ;
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.context,
         'n', 'n',
         numOutputPixels, bias.getNumElements(), 1,
         alpha,
         allOnesMemory, numOutputPixels,
         (type*)bias.getMemory(), 1,
         beta,
         (type*)output.getMemory() + outputOffset, numOutputPixels) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }

  done:
    return op.context.passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct ConvolutionBackward
{
  vl::ErrorCode operator()
  (Convolution &op,
   Tensor &derInput,
   Tensor &derFilter,
   Tensor &derBias,
   Tensor const &input,
   Tensor const &filter,
   Tensor const &derOutput)
  {

    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    ptrdiff_t numGroups = 0 ;
    ptrdiff_t numFiltersPerGroup = 0 ;
    ptrdiff_t filterVolume = 0 ;
    type const* allOnesMemory = NULL ;
    ptrdiff_t tempVolume = 0 ;
    type* tempMemory = NULL ;

    // for all derivatives
    assert(derOutput) ;
    ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;

    if (derBias) {
      // for derivative w.r.t. bias
      allOnesMemory = (type*) op.context.getAllOnes(deviceType,
                                                 dataType,
                                                 numOutputPixels) ;
      if (allOnesMemory == NULL) {
        error = op.context.getLastError() ;
        goto done ;
      }
    }

    if (derInput) {
      // for derivative w.r.t. data
      assert(filter) ;
      numGroups = derInput.getDepth() / filter.getDepth() ;
      filterVolume = filter.getHeight() * filter.getWidth() * filter.getDepth() ;
    }
    else if (derFilter) {
      // for derivative w.r.t. filter
      assert(input) ;
      numGroups = input.getDepth() / derFilter.getDepth() ;
      filterVolume = derFilter.getHeight() * derFilter.getWidth() * derFilter.getDepth() ;
    }
    numFiltersPerGroup = derOutput.getDepth() / numGroups ;

    // get scratch space
    tempVolume = numOutputPixels * filterVolume * numGroups ;
    if (tempVolume) {
      tempMemory = (type*) op.context.getWorkspace(deviceType, tempVolume * sizeof(type)) ;
      if (tempMemory == NULL) {
        error = op.context.getLastError() ;
        goto done ;
      }
    }

    for (int image = 0 ; image < derOutput.getSize() ; ++image) {

      ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

      /* compute derInput dz/dbias */
      if (derBias) {
        // has derBias, derOutput
        type alpha = 1 ;
        type beta = (image > 0) ; /* this saves init. the output array with 0 */
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.context,
         't',
         numOutputPixels, derOutput.getDepth(),
         alpha, /* alpha */
         (type const*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
         allOnesMemory, 1,
         beta, /* beta */
         (type*)derBias.getMemory(), 1) ;
        if (error != vl::VLE_Success) { return error ; }
      }

      /* compute derInpu dz/dx */
      if (derInput) {
        // has derInpu, derOutput, filter
        ptrdiff_t derInpuOffset = (derInput.getHeight()*derInput.getWidth()*derInput.getDepth()) * image ;
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
          ptrdiff_t tempGrpOffset = numOutputPixels * filterVolume * g ;
          ptrdiff_t derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
          type alpha = 1 ;
          type beta = 0 ;
          error = vl::impl::blas<deviceType,dataType>::gemm
          (op.context,
           'n', 't',
           numOutputPixels, filterVolume, numFiltersPerGroup,
           alpha,
           (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
           (type*)filter.getMemory() + filterGrpOffset, filterVolume,
           beta,
           tempMemory + tempGrpOffset, numOutputPixels) ;
          if (error != vl::VLE_Success) { return error ; }
        }
        error = vl::impl::im2row<deviceType,type>::backward
        (op.context,
         (type*)derInput.getMemory() + derInpuOffset,
         tempMemory,
         derInput.getHeight(), derInput.getWidth(), derInput.getDepth(),
         filter.getHeight(), filter.getWidth(),
         op.strideY, op.strideX,
         op.padTop, op.padBottom, op.padLeft, op.padRight,
         op.dilateY, op.dilateX) ;
        if (error != vl::VLE_Success) { return error ; }
      }

      /* compute derFilter dz/dF */
      if (derFilter) {
        // has derFilter, derOutput, data
        ptrdiff_t dataOffset = (input.getHeight()*input.getWidth()*input.getDepth()) * image ;
        error = vl::impl::im2row<deviceType,type>::forward
        (op.context,
         (type*)tempMemory,
         (type*)input.getMemory() + dataOffset,
         input.getHeight(), input.getWidth(), input.getDepth(),
         derFilter.getHeight(), derFilter.getWidth(),
         op.strideY, op.strideX,
         op.padTop, op.padBottom, op.padLeft, op.padRight,
         op.dilateY, op.dilateX) ;
        if (error != vl::VLE_Success) { return error ; }
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
          ptrdiff_t tempGrpOffset = numOutputPixels * filterVolume * g ;
          ptrdiff_t derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
          /* dzdF = temp' * dzdY */
          type alpha = 1 ;
          type beta = (image > 0) ; /* this saves init. the output array with 0 */
          error = vl::impl::blas<deviceType,dataType>::gemm
          (op.context,
           't', 'n',
           filterVolume, numFiltersPerGroup, numOutputPixels,
           alpha,
           tempMemory + tempGrpOffset, numOutputPixels,
           (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
           beta,
           (type*)derFilter.getMemory() + filterGrpOffset, filterVolume) ;
          if (error != vl::VLE_Success) { return error ; }
        }
      }
    }

  done:
    return op.context.passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                       Convolution Transpose Forward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct ConvolutionTransposeForward
{
  vl::ErrorCode operator()
  (ConvolutionTranspose &op,
   vl::Tensor &output,
   vl::Tensor const &input,
   vl::Tensor const &filter,
   vl::Tensor const &bias)
  {

    vl::ErrorCode error = VLE_Success ;
    size_t dataOffset = input.getHeight()*input.getWidth()*input.getDepth() ;
    size_t outputOffset = output.getHeight()*output.getWidth()*output.getDepth() ;

    // we need to process this down per image as nnconv_backward would otherwise
    // accumulate everything into a single feature field in the output
    for (int image = 0 ; image < input.getSize() ; ++image) {
      Tensor inputSlice(input) ;
      Tensor outputSlice(output) ;

      switch (input.getDataType()) {
        case VLDT_Float:
          inputSlice.setMemory((float*)input.getMemory() + dataOffset * image) ;
          outputSlice.setMemory((float*)output.getMemory() + outputOffset * image) ;
          break ;
        case VLDT_Double:
          inputSlice.setMemory((double*)input.getMemory() + dataOffset * image) ;
          outputSlice.setMemory((double*)output.getMemory() + outputOffset * image) ;
          break ;
        default:
          assert(false) ;
      }
      inputSlice.setSize(1) ;
      outputSlice.setSize(1) ;

      Convolution opc(op.context, op.upsampleY, op.upsampleX,
                      op.cropTop, op.cropBottom,
                      op.cropLeft, op.cropRight,
                      1, 1) ;
      Tensor null ;
      error = opc.backward(outputSlice, null, null,
                           null, filter, inputSlice) ;
      if (error != VLE_Success) { goto done ; }
    }
    if (bias) {
      error = vl::nn::Bias(op.context).forward(output,1.0,Tensor(),0,bias,1.0);
    }
  done:
    return error ;
  }
} ;

// -------------------------------------------------------------------
//                                      Convolution Transpose Backward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct ConvolutionTransposeBackward
{
  vl::ErrorCode operator()
  (ConvolutionTranspose &op,
   vl::Tensor &derInput,
   vl::Tensor &derFilter,
   vl::Tensor &derBias,
   vl::Tensor const &input,
   vl::Tensor const &filter,
   vl::Tensor const &derOutput)
  {
    vl::ErrorCode error = vl::VLE_Success ;
    Convolution opc(op.context,
                    op.upsampleY, op.upsampleX,
                    op.cropTop, op.cropBottom,
                    op.cropLeft, op.cropRight,
                    1, 1) ;
    Tensor null ;

    if (derInput) {
      error = opc.forward(derInput, 0,
                          derOutput, 1,
                          filter, null) ;
      if (error != VLE_Success) { goto done ; }
    }

    if (derFilter) {
      error = opc.backward(null, derFilter, null,
                           derOutput, Tensor(), input) ;
      if (error != VLE_Success) { goto done ; }
    }

    if (derBias) {
      Tensor null ;
      error = vl::nn::Bias(op.context).backward(null,0,derBias,0,0,1,derOutput) ;
    }

  done:
    return error ;
  }
} ;

// -------------------------------------------------------------------
//                                                             Drivers
// -------------------------------------------------------------------

#if ENABLE_CUDNN
#include "nnconv_cudnn.cu"
#endif

Convolution::Convolution(Context &context,
                         int strideY, int strideX,
                         int padTop, int padBottom,
                         int padLeft, int padRight,
                         int dilateY, int dilateX)
:
context(context),
strideY(strideY), strideX(strideX),
padTop(padTop), padBottom(padBottom),
padLeft(padLeft), padRight(padRight),
dilateY(dilateY), dilateX(dilateX)
{ }

vl::ErrorCode
Convolution::forward(Tensor &output, double outputMult,
                     Tensor const& input, double inputMult,
                     Tensor const& filter,
                     Tensor const& bias)
{
  return dispatch_cudnn<
  ConvolutionForward,
  ConvolutionForwardCudnn>()
  (*this,output,outputMult,input,inputMult,filter,bias) ;
}

vl::ErrorCode
Convolution::backward(Tensor &derInput,
                      Tensor &derFilter,
                      Tensor &derBias,
                      Tensor const &input,
                      Tensor const &filter,
                      Tensor const &derOutput)
{
  return dispatch_cudnn<
  ConvolutionBackward,
  ConvolutionBackwardCudnn>()
  (*this,derInput,derFilter,derBias,input,filter,derOutput) ;
}

ConvolutionTranspose::ConvolutionTranspose(Context &context,
                                           int upsampleY,
                                           int upsampleX,
                                           int cropTop,
                                           int cropBottom,
                                           int cropLeft,
                                           int cropRight)
:
context(context),
upsampleY(upsampleY),
upsampleX(upsampleX),
cropTop(cropTop),
cropBottom(cropBottom),
cropLeft(cropLeft),
cropRight(cropRight)
{ }

vl::ErrorCode
ConvolutionTranspose::forward(Tensor &output,
                              Tensor const& input,
                              Tensor const& filter,
                              Tensor const& bias)
{
  return dispatch<ConvolutionTransposeForward>()
  (*this,output,input,filter,bias) ;
}

vl::ErrorCode
ConvolutionTranspose::backward(Tensor &derInput,
                               Tensor &derFilter,
                               Tensor &derBias,
                               Tensor const &input,
                               Tensor const &filter,
                               Tensor const &derOutput)
{
  return dispatch<ConvolutionTransposeBackward>()
  (*this,derInput,derFilter,derBias,input,filter,derOutput) ;
}


