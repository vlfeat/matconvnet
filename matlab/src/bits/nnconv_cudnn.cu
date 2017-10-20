// @file nnconv_cudnn.cu
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnconv_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnconv.hpp"
#include "datacu.hpp"
#include "impl/cudnnhelper.hpp"
#include <cassert>
#include <algorithm>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = op.getContext().setError(op.getContext().getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__))) ; \
goto done ; \
} }

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
   Tensor const& filter,
   Tensor const& bias)
  {
    assert(output) ;
    assert(input) ;
    assert(filter) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, biasDesc, dataDesc ;
    cudnnFilterDescriptor_t filterDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool outputDescInitialized = false ;
    bool biasDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool filterDescInitialized = false ;
    bool convDescInitialized = false ;

    void* workSpace = NULL ;

    Int numGroups = input.getDepth() / filter.getDepth() ;
    Int numFiltersPerGroup = filter.getSize() / numGroups ;

    if (op.getDilation(1) != 1 || op.getDilation(0) != 1) return vl::VLE_Unsupported ;
    if (op.getPadding(2) != op.getPadding(3)) return vl::VLE_Unsupported ;
    if (op.getPadding(0) != op.getPadding(1)) return vl::VLE_Unsupported ;
    if (filter.getHeight() > input.getHeight()) return vl::VLE_Unsupported ;
    if (filter.getWidth() > input.getWidth()) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(outputDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       (int)output.getSize(), // sizes
                                       (int)numFiltersPerGroup,
                                       (int)output.getWidth(),
                                       (int)output.getHeight(),
                                       (int)(output.getHeight()*output.getWidth()*output.getDepth()), //strides
                                       (int)(output.getHeight()*output.getWidth()),
                                       (int)output.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       DataTypeToCudnn<dataType>::dataType,
                                       (int)input.getSize(),
                                       (int)(input.getDepth() / numGroups),
                                       (int)input.getWidth(),
                                       (int)input.getHeight(),
                                       (int)(input.getHeight()*input.getWidth()*input.getDepth()), //strides
                                       (int)(input.getHeight()*input.getWidth()),
                                       (int)input.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateFilterDescriptor(&filterDesc)) ;
    filterDescInitialized = true ;
    CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                     DataTypeToCudnn<dataType>::dataType,
                                     IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                     (int)numFiltersPerGroup,
                                     (int)filter.getDepth(),
                                     (int)filter.getWidth(),
                                     (int)filter.getHeight())) ;

    if (bias) {
      CHECK(cudnnCreateTensorDescriptor(&biasDesc)) ;
      biasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(biasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       1,
                                       (int)(bias.getNumElements() / numGroups),
                                       1,
                                       1)) ;
    }

    // Get convolution descriptor
    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                          (int)op.getPadding(2), (int)op.getPadding(0),
                                          (int)op.getStride(1), (int)op.getStride(0),
                                          1,1, // upscale
                                          CUDNN_CROSS_CORRELATION
                                          IF_CUDNN_GE6(COMMA DataTypeToCudnn<dataType>::dataType))) ;
    // Sanity check
#if 1
    {
      int n, c, h, w ;
      cudnnGetConvolution2dForwardOutputDim(convDesc,
                                            dataDesc,
                                            filterDesc,
                                            &n, &c, &w, &h) ;
      bool sane =
      output.getSize() == n &&
      numFiltersPerGroup == c &&
      output.getWidth() == w &&
      output.getHeight() == h ;
      assert(sane) ;
    }
#endif

    op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

    if (!op.getContext().getCudaHelper().cudnnConvolutionFwdSpecificAlgo) {
      // Determine algorithm automatically
      CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                                dataDesc,
                                                filterDesc,
                                                convDesc,
                                                outputDesc,
                                                op.getContext().getCudaHelper().cudnnConvolutionFwdPreference,
                                                op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceLimit,
                                                &op.getContext().getCudaHelper().cudnnConvolutionFwdAlgo)) ;
    }

    // Get workspace size
    CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  dataDesc,
                                                  filterDesc,
                                                  convDesc,
                                                  outputDesc,
                                                  op.getContext().getCudaHelper().cudnnConvolutionFwdAlgo,
                                                  &op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed)) ;

    // Get workspace
    if (op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed > 0) {
      workSpace = op.getContext().getWorkspace(vl::VLDT_GPU, op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed) ;
      if (workSpace == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }
    }

    // Perform convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      Int dataGrpOffset = (input.getHeight() * input.getWidth() * filter.getDepth()) *  g ;
      Int filterGrpOffset = (filter.getHeight() * filter.getWidth() * filter.getDepth()) * numFiltersPerGroup * g ;
      Int outputGrpOffset = (output.getHeight() * output.getWidth() * numFiltersPerGroup) * g ;
      Int biasGrpOffset = numFiltersPerGroup * g ;

      auto alpha = static_cast<type>(inputMult) ;
      auto beta = static_cast<type>(outputMult) ;
      CHECK(cudnnConvolutionForward(handle,
                                    &alpha,
                                    dataDesc, (type const*)input.getMemory() + dataGrpOffset,
                                    filterDesc, (type const*)filter.getMemory() + filterGrpOffset,
                                    convDesc,
                                    op.getContext().getCudaHelper().cudnnConvolutionFwdAlgo,
                                    workSpace, op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed,
                                    &beta,
                                    outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;

      if (bias) {
        type alpha = 1.0f ;
        type beta = 1.0f ;
#if (CUDNN_VERSION < 4000)
        CHECK(cudnnAddTensor(handle,
                             CUDNN_ADD_SAME_C,
                             &alpha,
                             biasDesc, (type const*)bias.getMemory() + biasGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#else
        CHECK(cudnnAddTensor(handle,
                             &alpha,
                             biasDesc, (type const*)bias.getMemory() + biasGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#endif
      }
    }

    /* cleanup */
  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filterDescInitialized) { cudnnDestroyFilterDescriptor(filterDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (biasDescInitialized) { cudnnDestroyTensorDescriptor(biasDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return op.getContext().passError(error, __func__) ;
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
   Tensor derBias,
   Tensor const &input,
   Tensor const &filter,
   Tensor const &derOutput)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derInputDesc needed as same as dataDesc */
    cudnnTensorDescriptor_t dataDesc, derBiasDesc, derOutputDesc ;
    cudnnFilterDescriptor_t filterDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool dataDescInitialized = false ;
    bool derBiasDescInitialized = false ;
    bool derOutputDescInitialized = false ;
    bool filterDescInitialized = false ;
    bool convDescInitialized = false ;

#if (CUDNN_VERSION >= 3000)
    void* workSpace = NULL ;
    size_t workSpaceSize = 0 ;
#endif

    Int numGroups = 1 ;
    Int numFiltersPerGroup = 0 ;
    Int filterVolume = 0 ;

    if (op.getDilation(1) != 1 || op.getDilation(0) != 1) return vl::VLE_Unsupported ;
    if (op.getPadding(2) != op.getPadding(3)) return vl::VLE_Unsupported ;
    if (op.getPadding(0) != op.getPadding(1)) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get the dimensions of the tensrors involved
    // If derInput is specified (hence comptued as output), use this
    // tensor as a basis to compute such dimensions, otherwise use derFilter.

    if (derInput) {
      assert(filter) ;
      numGroups = derInput.getDepth() / filter.getDepth() ;
      numFiltersPerGroup = filter.getSize() / numGroups ;
      filterVolume = filter.getHeight() * filter.getWidth() * filter.getDepth() ;

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                         DataTypeToCudnn<dataType>::dataType ,
                                         (int)derInput.getSize(),
                                         (int)(derInput.getDepth() / numGroups),
                                         (int)derInput.getWidth(),
                                         (int)derInput.getHeight(),
                                         (int)(derInput.getHeight()*derInput.getWidth()*derInput.getDepth()), //strides
                                         (int)(derInput.getHeight()*derInput.getWidth()),
                                         (int)derInput.getHeight(),
                                         1)) ;

      CHECK(cudnnCreateFilterDescriptor(&filterDesc)) ;
      filterDescInitialized = true ;
      CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       (int)numFiltersPerGroup,
                                       (int)filter.getDepth(),
                                       (int)filter.getWidth(),
                                       (int)filter.getHeight())) ;
    } else if (derFilter) {
      assert(input) ;
      numGroups = input.getDepth() / derFilter.getDepth() ;
      numFiltersPerGroup = derFilter.getSize() / numGroups ;
      filterVolume = derFilter.getHeight() * derFilter.getWidth() * derFilter.getDepth() ;

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                         DataTypeToCudnn<dataType>::dataType ,
                                         (int)input.getSize(),
                                         (int)(input.getDepth() / numGroups),
                                         (int)input.getWidth(),
                                         (int)input.getHeight(),
                                         (int)(input.getHeight()*input.getWidth()*input.getDepth()), //strides
                                         (int)(input.getHeight()*input.getWidth()),
                                         (int)(input.getHeight()),
                                         1)) ;

      CHECK(cudnnCreateFilterDescriptor(&filterDesc)) ;
      filterDescInitialized = true ;
      CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       (int)numFiltersPerGroup,
                                       (int)derFilter.getDepth(),
                                       (int)derFilter.getWidth(),
                                       (int)derFilter.getHeight())) ;
    }

    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                          (int)op.getPadding(2), (int)op.getPadding(0),
                                          (int)op.getStride(1), (int)op.getStride(0),
                                          1,1, // upscale
                                          CUDNN_CROSS_CORRELATION
                                          IF_CUDNN_GE6(COMMA DataTypeToCudnn<dataType>::dataType))) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(derOutputDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       (int)derOutput.getSize(), // sizes
                                       (int)numFiltersPerGroup,
                                       (int)derOutput.getWidth(),
                                       (int)derOutput.getHeight(),
                                       (int)(derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()), //strides
                                       (int)(derOutput.getHeight()*derOutput.getWidth()),
                                       (int)derOutput.getHeight(),
                                       1)) ;

    // for derivatives w.r.t. bias
    if (derBias) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasDesc)) ;
      derBiasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derBiasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       1,
                                       (int)(derBias.getNumElements() / numGroups),
                                       1,
                                       1)) ;
    }


    op.getContext().getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

#if (CUDNN_VERSION >= 3000)

    if (derFilter) {
      // Get filter derivatives algorithm
      CHECK(cudnnGetConvolutionBackwardFilterAlgorithm
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filterDesc,
             op.getContext().getCudaHelper().cudnnConvolutionBwdFilterPreference,
             op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceLimit,
             &op.getContext().getCudaHelper().cudnnConvolutionBwdFilterAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filterDesc,
             op.getContext().getCudaHelper().cudnnConvolutionBwdFilterAlgo,
             &op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, op.getContext().getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed) ;
    }

    if (derInput) {
      // Get data derivatives
      CHECK(cudnnGetConvolutionBackwardDataAlgorithm
            (handle,
             filterDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             op.getContext().getCudaHelper().cudnnConvolutionBwdDataPreference,
             op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceLimit,
             &op.getContext().getCudaHelper().cudnnConvolutionBwdDataAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize
            (handle,
             filterDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             op.getContext().getCudaHelper().cudnnConvolutionBwdDataAlgo,
             &op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, op.getContext().getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed) ;
    }

    // Get workspace
    if (workSpaceSize > 0) {
      workSpace = op.getContext().getWorkspace(vl::VLDT_GPU, workSpaceSize) ;
      if (workSpace == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }
    }
#endif

    // Perform backward convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      Int filterGrpOffset = filterVolume * numFiltersPerGroup  * g ;
      Int derOutputGrpOffset = (derOutput.getHeight() * derOutput.getWidth() * numFiltersPerGroup) * g ;

      if (derBias) {
        Int derBiasGrpOffset = numFiltersPerGroup * g ;
        type alpha = 1 ;
        type beta = 0 ;
        CHECK(cudnnConvolutionBackwardBias
              (handle,
               &alpha,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               &beta,
               derBiasDesc, (type*)derBias.getMemory() + derBiasGrpOffset)) ;
      }

      if (derFilter) {
        Int dataGrpOffset = (input.getHeight() * input.getWidth() * derFilter.getDepth()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;
#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardFilter)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardFilter_v3)
              (handle,
               &alpha,
               dataDesc, (type const*)input.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               op.getContext().getCudaHelper().cudnnConvolutionBwdFilterAlgo,
               workSpace, workSpaceSize,
               &beta,
               filterDesc, (type*)derFilter.getMemory() + filterGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardFilter
              (handle,
               &alpha,
               dataDesc, (type const*)input.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               filterDesc, (type*)derFilter.getMemory() + filterGrpOffset)) ;
#endif
      }

      if (derInput) {
        Int dataGrpOffset = (derInput.getHeight() * derInput.getWidth() * filter.getDepth()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;

#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardData)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardData_v3)
              (handle,
               &alpha,
               filterDesc, (type const*)filter.getMemory() + filterGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               op.getContext().getCudaHelper().cudnnConvolutionBwdDataAlgo,
               workSpace, workSpaceSize,
               &beta,
               dataDesc, (type*)derInput.getMemory() + dataGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardData
              (handle,
               &alpha,
               filterDesc, filter.getMemory() + filterGrpOffset,
               derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               dataDesc, derInput.getMemory() + dataGrpOffset)) ;
#endif
      }
    }

  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filterDescInitialized) { cudnnDestroyFilterDescriptor(filterDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    if (derBiasDescInitialized) { cudnnDestroyTensorDescriptor(derBiasDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    return op.getContext().passError(error, __func__) ;
  }

  } ;



