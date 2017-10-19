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
using namespace std ;

template<DeviceType deviceType, DataType dataType> struct ConvolutionForward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionBackward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionTransposeForward ;
template<DeviceType deviceType, DataType dataType> struct ConvolutionTransposeBackward ;
template<DataType dataType> struct ConvolutionForwardCudnn ;
template<DataType dataType> struct ConvolutionBackwardCudnn ;

// -------------------------------------------------------------------
/// MARK: - Convolution
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
    assert(input.getNumDimensions() <= 4) ; // Todo: generalize.

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    auto numGroups = as_signed(input.getDepth() / filter.getDepth()) ;
    auto numFiltersPerGroup = as_signed(filter.getSize()) / numGroups ;
    auto numOutputPixels = as_signed(output.getHeight() * output.getWidth()) ;
    auto filterVolume = as_signed(filter.getHeight() * filter.getWidth() * filter.getDepth()) ;
    auto tempVolume = numOutputPixels * filterVolume * numGroups ;

    type* tempMemory = (type*) op.getContext().getWorkspace
    (deviceType, as_unsigned(tempVolume) * sizeof(type)) ;

    type const* allOnesMemory = (type*) op.getContext().getAllOnes
    (deviceType, dataType, as_unsigned(numOutputPixels)) ;

    if (tempMemory == NULL || allOnesMemory == NULL) {
      error = op.getContext().getLastError() ;
      goto done ;
    }

    for (size_t image = 0 ; image < input.getSize() ; ++image) {

      auto dataOffset = (input.getHeight()*input.getWidth()*input.getDepth()) * image ;
      auto outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;

      error = vl::impl::im2row<deviceType,type>::forward
      (op.getContext(),
       tempMemory,
       (type*)input.getMemory() + dataOffset,
       input.getHeight(), input.getWidth(), input.getDepth(),
       filter.getHeight(), filter.getWidth(),
       as_unsigned(op.getStride(0)),
       as_unsigned(op.getStride(1)),
       as_unsigned(op.getPadding(0)),
       as_unsigned(op.getPadding(1)),
       as_unsigned(op.getPadding(2)),
       as_unsigned(op.getPadding(3)),
       op.getDilation(0), op.getDilation(1)) ;
      if (error != vl::VLE_Success) { goto done ; }

      for (Int g = 0 ; g < numGroups ; ++ g) {
        Int filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
        Int tempGrpOffset = numOutputPixels * filterVolume * g ;
        Int outputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        type alpha = inputMult ;
        type beta = outputMult ;
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
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
        (op.getContext(),
         'n', 'n',
         as_signed(numOutputPixels),
         as_signed(bias.getNumElements()), 1,
         alpha,
         allOnesMemory, numOutputPixels,
         (type*)bias.getMemory(), 1,
         beta,
         (type*)output.getMemory() + outputOffset, numOutputPixels) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }

  done:
    return op.getContext().passError(error, __func__) ;
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

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    Int numGroups = 0 ;
    Int numFiltersPerGroup = 0 ;
    Int filterVolume = 0 ;
    type const* allOnesMemory = NULL ;
    Int tempVolume = 0 ;
    type* tempMemory = NULL ;

    // for all derivatives
    assert(derOutput) ;
    Int numOutputPixels = as_signed(derOutput.getHeight() * derOutput.getWidth()) ;

    if (derBias) {
      // for derivative w.r.t. bias
      allOnesMemory = (type*) op.getContext().getAllOnes(deviceType,
                                                 dataType,
                                                 as_unsigned(numOutputPixels)) ;
      if (allOnesMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }
    }

    if (derInput) {
      // for derivative w.r.t. data
      assert(filter) ;
      numGroups = as_signed(derInput.getDepth() / filter.getDepth()) ;
      filterVolume = as_signed(filter.getHeight() * filter.getWidth() * filter.getDepth()) ;
    }
    else if (derFilter) {
      // for derivative w.r.t. filter
      assert(input) ;
      numGroups = as_signed(input.getDepth() / derFilter.getDepth()) ;
      filterVolume = as_signed(derFilter.getHeight() * derFilter.getWidth() * derFilter.getDepth()) ;
    }
    numFiltersPerGroup = as_signed(derOutput.getDepth()) / numGroups ;

    // get scratch space
    tempVolume = numOutputPixels * filterVolume * numGroups ;
    if (tempVolume) {
      tempMemory = (type*) op.getContext().getWorkspace(deviceType, as_unsigned(tempVolume) * sizeof(type)) ;
      if (tempMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }
    }

    for (Int image = 0 ; image < as_signed(derOutput.getSize()) ; ++image) {

      Int derOutputOffset = as_signed(derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

      /* compute derInput dz/dbias */
      if (derBias) {
        // has derBias, derOutput
        type alpha = 1 ;
        type beta = (image > 0) ; /* this saves init. the output array with 0 */
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.getContext(),
         't',
         numOutputPixels, as_signed(derOutput.getDepth()),
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
        Int derInpuOffset = as_signed(derInput.getHeight()*derInput.getWidth()*derInput.getDepth()) * image ;
        for (Int g = 0 ; g < numGroups ; ++ g) {
          Int filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
          Int tempGrpOffset = numOutputPixels * filterVolume * g ;
          Int derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
          type alpha = 1 ;
          type beta = 0 ;
          error = vl::impl::blas<deviceType,dataType>::gemm
          (op.getContext(),
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
        (op.getContext(),
         (type*)derInput.getMemory() + derInpuOffset,
         tempMemory,
         derInput.getHeight(), derInput.getWidth(), derInput.getDepth(),
         filter.getHeight(), filter.getWidth(),
         as_unsigned(op.getStride(0)),
         as_unsigned(op.getStride(1)),
         as_unsigned(op.getPadding(0)),
         as_unsigned(op.getPadding(1)),
         as_unsigned(op.getPadding(2)),
         as_unsigned(op.getPadding(3)),
         op.getDilation(0), op.getDilation(1)) ;
        if (error != vl::VLE_Success) { return error ; }
      }

      /* compute derFilter dz/dF */
      if (derFilter) {
        // has derFilter, derOutput, data
        Int dataOffset = as_signed(input.getHeight()*input.getWidth()*input.getDepth()) * image ;
        error = vl::impl::im2row<deviceType,type>::forward
        (op.getContext(),
         (type*)tempMemory,
         (type*)input.getMemory() + dataOffset,
         input.getHeight(), input.getWidth(), input.getDepth(),
         derFilter.getHeight(), derFilter.getWidth(),
         as_unsigned(op.getStride(0)),
         as_unsigned(op.getStride(1)),
         as_unsigned(op.getPadding(0)),
         as_unsigned(op.getPadding(1)),
         as_unsigned(op.getPadding(2)),
         as_unsigned(op.getPadding(3)),
         op.getDilation(0), op.getDilation(1)) ;
        if (error != vl::VLE_Success) { return error ; }
        for (Int g = 0 ; g < numGroups ; ++ g) {
          Int filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
          Int tempGrpOffset = numOutputPixels * filterVolume * g ;
          Int derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
          /* dzdF = temp' * dzdY */
          type alpha = 1 ;
          type beta = (image > 0) ; /* this saves init. the output array with 0 */
          error = vl::impl::blas<deviceType,dataType>::gemm
          (op.getContext(),
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
    return op.getContext().passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Convolution transpose
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
    Int dataOffset = as_signed(input.getHeight()*input.getWidth()*input.getDepth()) ;
    Int outputOffset = as_signed(output.getHeight()*output.getWidth()*output.getDepth()) ;

    // we need to process this down per image as nnconv_backward would otherwise
    // accumulate everything into a single feature field in the output
    for (Int image = 0 ; image < as_signed(input.getSize()) ; ++image) {
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

      Convolution opc(op.getContext(),
                      op.getUpsample(0), op.getUpsample(1),
                      op.getCrop(0),
                      op.getCrop(1),
                      op.getCrop(2),
                      op.getCrop(3),
                      1, 1) ;
      Tensor null ;
      error = opc.backward(outputSlice, null, null,
                           null, filter, inputSlice) ;
      if (error != VLE_Success) { goto done ; }
    }
    if (bias) {
      error = vl::nn::Bias(op.getContext()).forward(output,1.0,Tensor(),0,bias,1.0);
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
    Convolution opc(op.getContext(),
                    op.getUpsample(0), op.getUpsample(1),
                    op.getCrop(0),
                    op.getCrop(1),
                    op.getCrop(2),
                    op.getCrop(3),
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
      error = vl::nn::Bias(op.getContext()).backward(null,0,derBias,0,0,1,derOutput) ;
    }

  done:
    return error ;
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Drivers
// -------------------------------------------------------------------

#if ENABLE_CUDNN
#include "nnconv_cudnn.cu"
#endif

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
: ConvolutionLike(context)
{
  dilation.fill(1) ;
}

vl::ErrorCode
Convolution::setDilation(vector<Int> const& dilation)
{
  // There must one stride per spatial dimension.
  if (Int(dilation.size()) != getNumSpatialDimensions()) {
    return VLE_IllegalArgument ;
  }
  // Dilation must be positive.
  if (any_of(begin(dilation),begin(dilation)+getNumSpatialDimensions(),
             [](Int x){return x <= 0;})) {
    return VLE_IllegalArgument ;
  }
  copy(begin(dilation),begin(dilation)+getNumSpatialDimensions(),
       begin(this->dilation)) ;
  return VLE_Success ;
}

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
Convolution::forwardShape(TensorShape &output,
                          TensorShape const& input,
                          TensorShape const& filter)
{
  output = TensorShape() ; // null
  if (input.getNumDimensions() != filter.getNumDimensions()) {
    return VLE_IllegalArgument ;
  }
  if (as_signed(input.getNumDimensions()) < getNumSpatialDimensions()) {
    return VLE_IllegalArgument ;
  }
  output = input ;
  for (Int d = 0 ; d < getNumSpatialDimensions() ; ++d) {
    auto odim = convLikeSizeHelper(as_signed(input.getDimensions()[d]),
                                   as_signed(filter.getDimensions()[d]),
                                   getStride(d),
                                   {getPadding(2*d),getPadding(2*d+1)},
                                   getDilation(d)) ;
    output.setDimension(as_unsigned(d), as_unsigned(odim)) ;
  }
  return VLE_Success ;
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
                                           Int upsampleY,
                                           Int upsampleX,
                                           Int cropTop,
                                           Int cropBottom,
                                           Int cropLeft,
                                           Int cropRight)
:
Operation(context),
numSpatialDimensions(2),
upsample {upsampleY, upsampleX},
crop {cropTop, cropBottom, cropLeft, cropRight}
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


