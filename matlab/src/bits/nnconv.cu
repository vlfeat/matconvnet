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

    VLLOG(op,1)
    << "ConvolutionForward: BLAS, "
    << DeviceTypeTraits<deviceType>::name << ", "
    << DataTypeTraits<dataType>::name ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    Int numGroups = input.getNumChannels() / filter.getNumChannels() ;
    Int numFiltersPerGroup = filter.getCardinality() / numGroups ;
    Int numOutputPixels = output.getHeight() * output.getWidth() ;
    Int filterVolume = filter.getHeight() * filter.getWidth() * filter.getNumChannels() ;
    Int tempVolume = numOutputPixels * filterVolume * numGroups ;

    type* tempMemory = (type*) op.getContext().getWorkspace
    (deviceType, as_unsigned(tempVolume) * sizeof(type)) ;

    type const* allOnesMemory = (type*) op.getContext().getAllOnes
    (deviceType, dataType, as_unsigned(numOutputPixels)) ;

    if (tempMemory == NULL || allOnesMemory == NULL) {
      error = op.getContext().getLastError() ;
      goto done ;
    }

    for (Int image = 0 ; image < input.getCardinality() ; ++image) {

      auto dataOffset = (input.getHeight()*input.getWidth()*input.getNumChannels()) * image ;
      auto outputOffset = (output.getHeight()*output.getWidth()*output.getNumChannels()) * image ;

      error = vl::impl::im2row<deviceType,type>::forward
      (op.getContext(),
       tempMemory,
       (type*)input.getMemory() + dataOffset,
       input.getHeight(), input.getWidth(), input.getNumChannels(),
       filter.getHeight(), filter.getWidth(),
       op.getStride(0),
       op.getStride(1),
       op.getPadding(0),
       op.getPadding(1),
       op.getPadding(2),
       op.getPadding(3),
       op.getDilation(0),
       op.getDilation(1)) ;
      if (error != vl::VLE_Success) { goto done ; }

      for (Int g = 0 ; g < numGroups ; ++ g) {
        Int filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
        Int tempGrpOffset = numOutputPixels * filterVolume * g ;
        Int outputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        auto alpha = static_cast<type>(inputMult) ;
        auto beta = static_cast<type>(outputMult) ;
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
         numOutputPixels,
         bias.getNumElements(), 1,
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
    VLLOG(op,1)
    << "ConvolutionBackward: BLAS, "
    << DeviceTypeTraits<deviceType>::name << ", "
    << DataTypeTraits<dataType>::name ;

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
    Int numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;

    if (derBias) {
      // for derivative w.r.t. bias
      allOnesMemory = (type*) op.getContext().getAllOnes
      (deviceType,
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
      numGroups = derInput.getNumChannels() / filter.getNumChannels() ;
      filterVolume = filter.getHeight() * filter.getWidth() * filter.getNumChannels() ;
    }
    else if (derFilter) {
      // for derivative w.r.t. filter
      assert(input) ;
      numGroups = input.getNumChannels() / derFilter.getNumChannels() ;
      filterVolume = derFilter.getHeight() * derFilter.getWidth() * derFilter.getNumChannels() ;
    }
    numFiltersPerGroup = derOutput.getNumChannels() / numGroups ;

    // get scratch space
    tempVolume = numOutputPixels * filterVolume * numGroups ;
    if (tempVolume) {
      tempMemory = (type*) op.getContext().getWorkspace(deviceType, as_unsigned(tempVolume) * sizeof(type)) ;
      if (tempMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }
    }

    for (Int image = 0 ; image < derOutput.getCardinality() ; ++image) {

      Int derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getNumChannels()) * image ;

      /* compute derInput dz/dbias */
      if (derBias) {
        // has derBias, derOutput
        type alpha = 1 ;
        type beta = (image > 0) ; /* this saves init. the output array with 0 */
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.getContext(),
         't',
         numOutputPixels, derOutput.getNumChannels(),
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
        Int derInpuOffset = (derInput.getHeight()*derInput.getWidth()*derInput.getNumChannels()) * image ;
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
         derInput.getHeight(), derInput.getWidth(), derInput.getNumChannels(),
         filter.getHeight(), filter.getWidth(),
         op.getStride(0),
         op.getStride(1),
         op.getPadding(0),
         op.getPadding(1),
         op.getPadding(2),
         op.getPadding(3),
         op.getDilation(0),
         op.getDilation(1)) ;
        if (error != vl::VLE_Success) { return error ; }
      }

      /* compute derFilter dz/dF */
      if (derFilter) {
        // has derFilter, derOutput, data
        Int dataOffset = (input.getHeight()*input.getWidth()*input.getNumChannels()) * image ;
        error = vl::impl::im2row<deviceType,type>::forward
        (op.getContext(),
         (type*)tempMemory,
         (type*)input.getMemory() + dataOffset,
         input.getHeight(), input.getWidth(), input.getNumChannels(),
         derFilter.getHeight(), derFilter.getWidth(),
         op.getStride(0),
         op.getStride(1),
         op.getPadding(0),
         op.getPadding(1),
         op.getPadding(2),
         op.getPadding(3),
         op.getDilation(0),
         op.getDilation(1)) ;
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
    Int dataOffset = input.getHeight()*input.getWidth()*input.getNumChannels() ;
    Int outputOffset = output.getHeight()*output.getWidth()*output.getNumChannels() ;

    // we need to process this down per image as nnconv_backward would otherwise
    // accumulate everything into a single feature field in the output
    for (Int image = 0 ; image < input.getCardinality() ; ++image) {
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
  // Dilation must be positive.
  if (any_of(begin(dilation),end(dilation),[](Int x){return x <= 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "Convolution: a dilation parameter is not positive.") ;
  }

  // There must one dilation per spatial dimension.
  if (Int(dilation.size()) == getNumSpatialDimensions()) {
    copy(begin(dilation),end(dilation),begin(this->dilation)) ;
  }
  else if (dilation.size() == 1) {
    this->dilation.fill(dilation[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "DILATION is neither scalar nor has the same cardinality as the number of spatial dimensions.") ;
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
      output.setDimension(d,odim) ;
    }
    output.setDimension(ns, numFilters) ;
    output.setDimension(ns+1, input.getDimension(ns+1)) ;
  } else {
    // Bias / subsample mode.
    output = input ;
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
  }

  // Check the bias.
  bool hasBias = !bias.isEmpty() ;
  if (hasBias) {
    if (bias.getNumElements() != output.getNumChannels()) {
      output.clear() ;
      return  getContext().setError
      (VLE_TensorShapeMismatch,
       "Convolution: BIAS does not have a number of element equal to the number of output feature channels.") ;
    }
  }

  return VLE_Success ;
}

vl::ErrorCode
Convolution::forward(Tensor &output, double outputMult,
                     Tensor const& input, double inputMult,
                     Tensor const& filter,
                     Tensor const& bias)
{
  ErrorCode error ;

  // Validate arguments.
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape, input, filter, bias)) != VLE_Success) {
    return error ;
  }
  if (output != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "Convolution: OUTPUT does not have the appropriate dimensions.") ;
  }
  if (!input.isEmpty() && input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "Convolution: INPUT is not an empty tensor, but it has no data either.") ;
  }
  if (!output.isEmpty() && output.isNull()) {
    return  getContext().setError
    (VLE_IllegalArgument,
     "Convolution: OUTPUT is not an empty tensor, but it has no data either.") ;
  }
  if (!filter.isEmpty() && filter.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "Convolution: FILTER is not an empty tensor, but it has no data either.") ;
  }
  if (!bias.isNull() && bias.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "Convolution: BIAS is not an empty tensor, but it has no data either.") ;
  }
  if (!check_tensor_compatibility(output,input,filter,bias)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "Convolution: the tensors have mismatching data or device type.") ;
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

  // Do the work.
  return dispatch_cudnn<
  ConvolutionForward,
  ConvolutionForwardCudnn>()
  (*this,output,outputMult,input,inputMult,filter,bias) ;
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
                      Tensor const &derOutput)
{
  ErrorCode error ;

  // Check that all tensors have the same type.
  if (!check_tensor_compatibility(derInput,derFilter,derBias,input,filter,derOutput)) {
    return getContext().passError(VLE_IllegalArgument, "ConvolutionBackward:") ;
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

  // If the bias derivaties are requested, check that we have what we need.
  if (!derBias.isEmpty()) {
    if (derBias.isNull()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERBIAS requested, but the output tensor is null.") ;
    }
    if (derBias.getNumElements() != input.getNumChannels()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "ConvolutionBackward: DERBIAS requested, but it has an incorrect size.") ;
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


