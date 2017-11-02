// @file nnconv_cudnn.cu
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnconv_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "../datacu.hpp"
#include "cudnnhelper.hpp"
#include <cassert>
#include <algorithm>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct ConvolutionForwardCudnn
{
  vl::ErrorCode operator()
  (Convolution const &op,
   Tensor output, double outputMult,
   Tensor const& input, double inputMult,
   Tensor const& filter)
  {
    assert(output) ;
    assert(input) ;
    assert(filter) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnFilterDescriptor_t filterDesc ;
    cudnnConvolutionDescriptor_t convDesc ;

    void* workSpace = NULL ;

    Int numGroups = input.getNumChannels() / filter.getNumChannels() ;
    Int numFiltersPerGroup = filter.getCardinality() / numGroups ;

    if (op.getDilation(1) != 1 || op.getDilation(0) != 1) return vl::VLE_Unsupported ;
    if (op.getPadding(2) != op.getPadding(3)) return vl::VLE_Unsupported ;
    if (op.getPadding(0) != op.getPadding(1)) return vl::VLE_Unsupported ;
    if (filter.getHeight() > input.getHeight()) return vl::VLE_Unsupported ;
    if (filter.getWidth() > input.getWidth()) return vl::VLE_Unsupported ;

    static const std::string signature = std::string("ConvolutionForward[CuDNN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    // Get CuDNN.
    cudnnHandle_t handle ;
    CKCUDNN(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs.
    CudnnTensorDescriptor outputDesc, dataDesc ;
#if CUDNN_VERSION < 7000
    {
      TensorShape sliceShape = output.getShape() ;
      sliceShape.setDimension(2,numFiltersPerGroup) ;
      CKCUDNN(outputDesc.init(dataType,sliceShape)) ;
    }
    {
      TensorShape sliceShape = input.getShape() ;
      sliceShape.setDimension(2,input.getNumChannels() / numGroups) ;
      CKCUDNN(dataDesc.init(dataType,sliceShape)) ;
    }
#else
    CKCUDNN(outputDesc.init(dataType, output.getShape())) ;
    CKCUDNN(dataDesc.init(dataType, input.getShape())) ;
#endif

    CKCUDNN(cudnnCreateFilterDescriptor(&filterDesc)) ;
    auto filterDescDeleter = defer([&]{cudnnDestroyFilterDescriptor(filterDesc);}) ;
    CKCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                       DataTypeToCudnn<dataType>::dataType,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
#if CUDNN_VERSION < 7000
                                       (int)numFiltersPerGroup,
#else
                                       (int)filter.getCardinality(),
#endif
                                       (int)filter.getNumChannels(),
                                       (int)filter.getWidth(),
                                       (int)filter.getHeight())) ;

    // Get convolution descriptor.
    CKCUDNN(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    auto convDescDeleter = defer([&]{cudnnDestroyConvolutionDescriptor(convDesc);}) ;
    CKCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                            (int)op.getPadding(2), (int)op.getPadding(0),
                                            (int)op.getStride(1), (int)op.getStride(0),
                                            1,1, // upscale
                                            CUDNN_CROSS_CORRELATION
                                            IF_CUDNN_GE6(COMMA DataTypeToCudnn<dataType>::dataType))) ;

#if (CUDNN_VERSION >= 7000)
    CKCUDNN(cudnnSetConvolutionGroupCount(convDesc, (int)numGroups)) ;
#endif

    // Sanity check
#if 1
    {
      int n, c, h, w ;
      cudnnGetConvolution2dForwardOutputDim(convDesc,
                                            dataDesc.get(),
                                            filterDesc,
                                            &n, &c, &w, &h) ;
      bool sane =
      output.getCardinality() == n &&
#if (CUDNN_VERSION < 7000)
      numFiltersPerGroup == c &&
#else
      output.getNumChannels() == c &&
#endif
      output.getWidth() == w &&
      output.getHeight() == h ;
      assert(sane) ;
    }
#endif

    op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

    if (!op.getContext().getCudaHelper().cudnnConvolutionFwdSpecificAlgo) {
      // Determine algorithm automatically.
      CKCUDNN(cudnnGetConvolutionForwardAlgorithm(handle,
                                                  dataDesc.get(),
                                                  filterDesc,
                                                  convDesc,
                                                  outputDesc.get(),
                                                  op.getContext().getCudaHelper().cudnnConvolutionFwdPreference,
                                                  op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceLimit,
                                                  &op.getContext().getCudaHelper().cudnnConvolutionFwdAlgo)) ;
    }

    // Get workspace size.
    CKCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                    dataDesc.get(),
                                                    filterDesc,
                                                    convDesc,
                                                    outputDesc.get(),
                                                    op.getContext().getCudaHelper().cudnnConvolutionFwdAlgo,
                                                    &op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed)) ;

    // Get workspace.
    if (op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed > 0) {
      workSpace = op.getContext().getWorkspace(vl::VLDT_GPU, op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed) ;
      if (workSpace == NULL) {
        return op.getContext().setError
        (VLE_OutOfMemory, "ConvolutionForwardCudnn: out of memory.") ; // todo: signature
      }
    }

    // Perform convolution.
#if (CUDNN_VERSION < 7000)
    for (int g = 0  ; g < numGroups ; ++g) {
      Int dataGrpOffset = (input.getHeight() * input.getWidth() * filter.getNumChannels()) *  g ;
      Int filterGrpOffset = (filter.getHeight() * filter.getWidth() * filter.getNumChannels()) * numFiltersPerGroup * g ;
      Int outputGrpOffset = (output.getHeight() * output.getWidth() * numFiltersPerGroup) * g ;

      auto alpha = static_cast<type>(inputMult) ;
      auto beta = static_cast<type>(outputMult) ;
      CKCUDNN(cudnnConvolutionForward(handle,
                                      &alpha,
                                      dataDesc.get(), (type const*)input.getMemory() + dataGrpOffset,
                                      filterDesc, (type const*)filter.getMemory() + filterGrpOffset,
                                      convDesc,
                                      op.getContext().getCudaHelper().cudnnConvolutionFwdAlgo,
                                      workSpace, op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed,
                                      &beta,
                                      outputDesc.get(), (type*)output.getMemory() + outputGrpOffset)) ;
    }
#else
    {
      auto alpha = static_cast<type>(inputMult) ;
      auto beta = static_cast<type>(outputMult) ;
      CKCUDNN(cudnnConvolutionForward(handle,
                                      &alpha,
                                      dataDesc.get(), (type const*)input.getMemory(),
                                      filterDesc, (type const*)filter.getMemory(),
                                      convDesc,
                                      op.getContext().getCudaHelper().cudnnConvolutionFwdAlgo,
                                      workSpace, op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed,
                                      &beta,
                                      outputDesc.get(), (type*)output.getMemory())) ;
    }
#endif
    return VLE_Success ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct ConvolutionBackwardCudnn
{
  vl::ErrorCode operator()
  (Convolution const &op,
   Tensor derInput,
   Tensor derFilter,
   Tensor const &input,
   Tensor const &filter,
   Tensor const &derOutput)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    // Make descriptors.
    cudnnFilterDescriptor_t filterDesc ;
    CKCUDNN(cudnnCreateFilterDescriptor(&filterDesc)) ;
    auto filterDescDeleter = defer([&]{cudnnDestroyFilterDescriptor(filterDesc);}) ;

    cudnnConvolutionDescriptor_t convDesc ;
    CKCUDNN(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    auto convDescDeleter = defer([&]{cudnnDestroyConvolutionDescriptor(convDesc);}) ;

#if (CUDNN_VERSION >= 3000)
    void* workSpace = NULL ;
    size_t workSpaceSize = 0 ;
#endif

    Int numGroups = 1 ;
#if CUDNN_VERSION < 7000
    Int numFiltersPerGroup = 0 ;
    Int filterVolume = 0 ;
#endif

    if (op.getDilation(1) != 1 || op.getDilation(0) != 1) return vl::VLE_Unsupported ;
    if (op.getPadding(2) != op.getPadding(3)) return vl::VLE_Unsupported ;
    if (op.getPadding(0) != op.getPadding(1)) return vl::VLE_Unsupported ;

    static const std::string signature = std::string("ConvolutionBackward[CuDNN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    // Get CuDNN
    cudnnHandle_t handle ;
    CKCUDNN(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get the dimensions of the tensrors involved
    // If derInput is specified (hence comptued as output), use this
    // tensor as a basis to compute such dimensions, otherwise use derFilter.
    CudnnTensorDescriptor dataDesc ;

    if (derInput) {
      assert(filter) ;
      numGroups = derInput.getNumChannels() / filter.getNumChannels() ;
#if CUDNN_VERSION < 7000
      numFiltersPerGroup = filter.getCardinality() / numGroups ;
      filterVolume = filter.getHeight() * filter.getWidth() * filter.getNumChannels() ;
      {
        TensorShape sliceShape = derInput.getShape() ;
        sliceShape.setDimension(2,derInput.getNumChannels() / numGroups) ;
        CKCUDNN(dataDesc.init(dataType,sliceShape)) ;
      }
#else
      CKCUDNN(dataDesc.init(dataType,derInput.getShape())) ;
#endif
      CKCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                         DataTypeToCudnn<dataType>::dataType ,
                                         IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
#if CUDNN_VERSION < 7000
                                         (int)numFiltersPerGroup,
#else
                                         (int)filter.getCardinality(),
#endif
                                         (int)filter.getNumChannels(),
                                         (int)filter.getWidth(),
                                         (int)filter.getHeight())) ;
    } else if (derFilter) {
      assert(input) ;
      numGroups = input.getNumChannels() / derFilter.getNumChannels() ;
#if CUDNN_VERSION < 7000
      numFiltersPerGroup = derFilter.getCardinality() / numGroups ;
      filterVolume = derFilter.getHeight() * derFilter.getWidth() * derFilter.getNumChannels() ;
      {
        TensorShape sliceShape = input.getShape() ;
        sliceShape.setDimension(2,input.getNumChannels() / numGroups) ;
        CKCUDNN(outputDesc.init(dataType,sliceShape)) ;
      }
#else
      CKCUDNN(dataDesc.init(dataType,input.getShape())) ;
#endif
      CKCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                         DataTypeToCudnn<dataType>::dataType ,
                                         IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
#if CUDNN_VERSION < 7000
                                         (int)numFiltersPerGroup,
#else
                                         (int)derFilter.getCardinality(),
#endif
                                         (int)derFilter.getNumChannels(),
                                         (int)derFilter.getWidth(),
                                         (int)derFilter.getHeight())) ;
    }
    CKCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                            (int)op.getPadding(2), (int)op.getPadding(0),
                                            (int)op.getStride(1), (int)op.getStride(0),
                                            1,1, // upscale
                                            CUDNN_CROSS_CORRELATION
                                            IF_CUDNN_GE6(COMMA DataTypeToCudnn<dataType>::dataType))) ;

#if (CUDNN_VERSION >= 7000)
    CKCUDNN(cudnnSetConvolutionGroupCount(convDesc, (int)numGroups)) ;
#endif

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CudnnTensorDescriptor derOutputDesc ;
#if CUDNN_VERSION < 7000
    numFiltersPerGroup = derFilter.getCardinality() / numGroups ;
    filterVolume = derFilter.getHeight() * derFilter.getWidth() * derFilter.getNumChannels() ;
    {
      TensorShape sliceShape = derOutput.getShape() ;
      sliceShape.setDimension(2,numFiltersPerGroup) ;
      CKCUDNN(derOutputDesc.init(dataType,sliceShape)) ;
    }
#else
    CKCUDNN(derOutputDesc.init(dataType,derOutput.getShape())) ;
#endif

    op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

#if (CUDNN_VERSION >= 3000)

    if (derFilter) {
      // Get filter derivatives algorithm
      CKCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm
              (handle,
               dataDesc.get(),
               derOutputDesc.get(),
               convDesc,
               filterDesc,
               op.getContext().getCudaHelper().cudnnConvolutionBwdFilterPreference,
               op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceLimit,
               &op.getContext().getCudaHelper().cudnnConvolutionBwdFilterAlgo)) ;

      // Get workspace size
      CKCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize
              (handle,
               dataDesc.get(),
               derOutputDesc.get(),
               convDesc,
               filterDesc,
               op.getContext().getCudaHelper().cudnnConvolutionBwdFilterAlgo,
               &op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed) ;
    }

    if (derInput) {
      // Get data derivatives algorithm
      CKCUDNN(cudnnGetConvolutionBackwardDataAlgorithm
              (handle,
               filterDesc,
               derOutputDesc.get(),
               convDesc,
               dataDesc.get(),
               op.getContext().getCudaHelper().cudnnConvolutionBwdDataPreference,
               op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceLimit,
               &op.getContext().getCudaHelper().cudnnConvolutionBwdDataAlgo)) ;

      // Get workspace size
      CKCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize
              (handle,
               filterDesc,
               derOutputDesc.get(),
               convDesc,
               dataDesc.get(),
               op.getContext().getCudaHelper().cudnnConvolutionBwdDataAlgo,
               &op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed) ;
    }

    // Get workspace
    if (workSpaceSize > 0) {
      workSpace = op.getContext().getWorkspace(vl::VLDT_GPU, workSpaceSize) ;
      if (workSpace == NULL) {
        return op.getContext().setError
        (VLE_OutOfMemory, "ConvolutionBackwardCudnn: out of memory.") ; // todo: signature
      }
    }
#endif

    // Perform backward convolution for each filter group
#if CUDNN_VERSION < 7000
    for (int g = 0  ; g < numGroups ; ++g) {
      Int filterGrpOffset = filterVolume * numFiltersPerGroup  * g ;
      Int derOutputGrpOffset = (derOutput.getHeight() * derOutput.getWidth() * numFiltersPerGroup) * g ;

      if (derFilter) {
        Int dataGrpOffset = (input.getHeight() * input.getWidth() * derFilter.getNumChannels()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;
#if (CUDNN_VERSION >= 3000)
        CKCUDNN(IF_CUDNN_GE4(cudnnConvolutionBackwardFilter)
                IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardFilter_v3)
                (handle,
                 &alpha,
                 dataDesc.get(), (type const*)input.getMemory() + dataGrpOffset,
                 derOutputDesc.get(), (type const*)derOutput.getMemory() + derOutputGrpOffset,
                 convDesc,
                 op.getContext().getCudaHelper().cudnnConvolutionBwdFilterAlgo,
                 workSpace, workSpaceSize,
                 &beta,
                 filterDesc, (type*)derFilter.getMemory() + filterGrpOffset)) ;
#else
        CKCUDNN(cudnnConvolutionBackwardFilter
                (handle,
                 &alpha,
                 dataDesc.get(), (type const*)input.getMemory() + dataGrpOffset,
                 derOutputDesc.get(), (type const*)derOutput.getMemory() + derOutputGrpOffset,
                 convDesc,
                 &beta,
                 filterDesc, (type*)derFilter.getMemory() + filterGrpOffset)) ;
#endif
      }

      if (derInput) {
        Int dataGrpOffset = (derInput.getHeight() * derInput.getWidth() * filter.getNumChannels()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;

#if (CUDNN_VERSION >= 3000)
        CKCUDNN(IF_CUDNN_GE4(cudnnConvolutionBackwardData)
                IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardData_v3)
                (handle,
                 &alpha,
                 filterDesc, (type const*)filter.getMemory() + filterGrpOffset,
                 derOutputDesc.get(), (type const*)derOutput.getMemory() + derOutputGrpOffset,
                 convDesc,
                 op.getContext().getCudaHelper().cudnnConvolutionBwdDataAlgo,
                 workSpace, workSpaceSize,
                 &beta,
                 dataDesc.get(), (type*)derInput.getMemory() + dataGrpOffset)) ;
#else
        CKCUDNN(cudnnConvolutionBackwardData
                (handle,
                 &alpha,
                 filterDesc, filter.getMemory() + filterGrpOffset,
                 derOutputDesc.get(), derOutput.getMemory() + derOutputGrpOffset,
                 convDesc,
                 &beta,
                 dataDesc.get(), derInput.getMemory() + dataGrpOffset)) ;
#endif
      }
    }
#else // CUDNN >= 7.0
    if (derFilter) {
      type alpha = 1 ;
      type beta = 0 ;
      CKCUDNN(cudnnConvolutionBackwardFilter
              (handle,
               &alpha,
               dataDesc.get(), (type const*)input.getMemory(),
               derOutputDesc.get(), (type const*)derOutput.getMemory(),
               convDesc,
               op.getContext().getCudaHelper().cudnnConvolutionBwdFilterAlgo,
               workSpace, workSpaceSize,
               &beta,
               filterDesc, (type*)derFilter.getMemory())) ;
    }

    if (derInput) {
      type alpha = 1 ;
      type beta = 0 ;
      CKCUDNN(cudnnConvolutionBackwardData
              (handle,
               &alpha,
               filterDesc, (type const*)filter.getMemory(),
               derOutputDesc.get(), (type const*)derOutput.getMemory(),
               convDesc,
               op.getContext().getCudaHelper().cudnnConvolutionBwdDataAlgo,
               workSpace, workSpaceSize,
               &beta,
               dataDesc.get(), (type*)derInput.getMemory())) ;
    }
#endif
    return VLE_Success ;
  }
} ;



