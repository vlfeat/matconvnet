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
error = op.context.setError(op.context.getCudaHelper().catchCudnnError(cudnnError, \
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
  (Convolution &op,
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

    int numGroups = input.getDepth() / filter.getDepth() ;
    int numFiltersPerGroup = filter.getSize() / numGroups ;

    if (op.dilateX != 1 || op.dilateY != 1) return vl::VLE_Unsupported ;
    if (op.padLeft != op.padRight) return vl::VLE_Unsupported ;
    if (op.padTop != op.padBottom) return vl::VLE_Unsupported ;
    if (filter.getHeight() > input.getHeight()) return vl::VLE_Unsupported ;
    if (filter.getWidth() > input.getWidth()) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(outputDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       output.getSize(), // sizes
                                       numFiltersPerGroup,
                                       output.getWidth(),
                                       output.getHeight(),
                                       output.getHeight()*output.getWidth()*output.getDepth(), //strides
                                       output.getHeight()*output.getWidth(),
                                       output.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       DataTypeToCudnn<dataType>::dataType,
                                       input.getSize(),
                                       input.getDepth() / numGroups,
                                       input.getWidth(),
                                       input.getHeight(),
                                       input.getHeight()*input.getWidth()*input.getDepth(), //strides
                                       input.getHeight()*input.getWidth(),
                                       input.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateFilterDescriptor(&filterDesc)) ;
    filterDescInitialized = true ;
    CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                     DataTypeToCudnn<dataType>::dataType,
                                     IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                     numFiltersPerGroup,
                                     filter.getDepth(),
                                     filter.getWidth(),
                                     filter.getHeight())) ;

    if (bias) {
      CHECK(cudnnCreateTensorDescriptor(&biasDesc)) ;
      biasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(biasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       1,
                                       bias.getNumElements() / numGroups,
                                       1,
                                       1)) ;
    }

    // Get convolution descriptor
    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                          op.padLeft, op.padTop,
                                          op.strideX, op.strideY,
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

    op.context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    op.context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    op.context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

    if (!op.context.getCudaHelper().cudnnConvolutionFwdSpecificAlgo) {
      // Determine algorithm automatically
      CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                                dataDesc,
                                                filterDesc,
                                                convDesc,
                                                outputDesc,
                                                op.context.getCudaHelper().cudnnConvolutionFwdPreference,
                                                op.context.getCudaHelper().cudnnConvolutionFwdWorkSpaceLimit,
                                                &op.context.getCudaHelper().cudnnConvolutionFwdAlgo)) ;
    }

    // Get workspace size
    CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  dataDesc,
                                                  filterDesc,
                                                  convDesc,
                                                  outputDesc,
                                                  op.context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                                  &op.context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed)) ;

    // Get workspace
    if (op.context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed > 0) {
      workSpace = op.context.getWorkspace(vl::VLDT_GPU, op.context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed) ;
      if (workSpace == NULL) {
        error = op.context.getLastError() ;
        goto done ;
      }
    }

    // Perform convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t dataGrpOffset = (input.getHeight() * input.getWidth() * filter.getDepth()) *  g ;
      ptrdiff_t filterGrpOffset = (filter.getHeight() * filter.getWidth() * filter.getDepth()) * numFiltersPerGroup * g ;
      ptrdiff_t outputGrpOffset = (output.getHeight() * output.getWidth() * numFiltersPerGroup) * g ;
      ptrdiff_t biasGrpOffset = numFiltersPerGroup * g ;

      type alpha = inputMult ;
      type beta = outputMult ;
      CHECK(cudnnConvolutionForward(handle,
                                    &alpha,
                                    dataDesc, (type const*)input.getMemory() + dataGrpOffset,
                                    filterDesc, (type const*)filter.getMemory() + filterGrpOffset,
                                    convDesc,
                                    op.context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                    workSpace, op.context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed,
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
    return op.context.passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct ConvolutionBackwardCudnn
{
  vl::ErrorCode operator()
  (Convolution &op,
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

    ptrdiff_t numGroups = 1 ;
    ptrdiff_t numFiltersPerGroup = 0 ;
    ptrdiff_t filterVolume = 0 ;

    if (op.dilateX != 1 || op.dilateY != 1) return vl::VLE_Unsupported ;
    if (op.padLeft != op.padRight) return vl::VLE_Unsupported ;
    if (op.padTop != op.padBottom) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

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
                                         derInput.getSize(),
                                         derInput.getDepth() / numGroups,
                                         derInput.getWidth(),
                                         derInput.getHeight(),
                                         derInput.getHeight()*derInput.getWidth()*derInput.getDepth(), //strides
                                         derInput.getHeight()*derInput.getWidth(),
                                         derInput.getHeight(),
                                         1)) ;

      CHECK(cudnnCreateFilterDescriptor(&filterDesc)) ;
      filterDescInitialized = true ;
      CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       numFiltersPerGroup,
                                       filter.getDepth(),
                                       filter.getWidth(),
                                       filter.getHeight())) ;
    } else if (derFilter) {
      assert(input) ;
      numGroups = input.getDepth() / derFilter.getDepth() ;
      numFiltersPerGroup = derFilter.getSize() / numGroups ;
      filterVolume = derFilter.getHeight() * derFilter.getWidth() * derFilter.getDepth() ;

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                         DataTypeToCudnn<dataType>::dataType ,
                                         input.getSize(),
                                         input.getDepth() / numGroups,
                                         input.getWidth(),
                                         input.getHeight(),
                                         input.getHeight()*input.getWidth()*input.getDepth(), //strides
                                         input.getHeight()*input.getWidth(),
                                         input.getHeight(),
                                         1)) ;

      CHECK(cudnnCreateFilterDescriptor(&filterDesc)) ;
      filterDescInitialized = true ;
      CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       numFiltersPerGroup,
                                       derFilter.getDepth(),
                                       derFilter.getWidth(),
                                       derFilter.getHeight())) ;
    }

    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                          op.padLeft, op.padTop,
                                          op.strideX, op.strideY,
                                          1,1, // upscale
                                          CUDNN_CROSS_CORRELATION
                                          IF_CUDNN_GE6(COMMA DataTypeToCudnn<dataType>::dataType))) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(derOutputDesc,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       derOutput.getSize(), // sizes
                                       numFiltersPerGroup,
                                       derOutput.getWidth(),
                                       derOutput.getHeight(),
                                       derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth(), //strides
                                       derOutput.getHeight()*derOutput.getWidth(),
                                       derOutput.getHeight(),
                                       1)) ;

    // for derivatives w.r.t. bias
    if (derBias) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasDesc)) ;
      derBiasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derBiasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType ,
                                       1,
                                       derBias.getNumElements() / numGroups,
                                       1,
                                       1)) ;
    }


    op.context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    op.context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    op.context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

#if (CUDNN_VERSION >= 3000)

    if (derFilter) {
      // Get filter derivatives algorithm
      CHECK(cudnnGetConvolutionBackwardFilterAlgorithm
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filterDesc,
             op.context.getCudaHelper().cudnnConvolutionBwdFilterPreference,
             op.context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceLimit,
             &op.context.getCudaHelper().cudnnConvolutionBwdFilterAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filterDesc,
             op.context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
             &op.context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, op.context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed) ;
    }

    if (derInput) {
      // Get data derivatives
      CHECK(cudnnGetConvolutionBackwardDataAlgorithm
            (handle,
             filterDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             op.context.getCudaHelper().cudnnConvolutionBwdDataPreference,
             op.context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceLimit,
             &op.context.getCudaHelper().cudnnConvolutionBwdDataAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize
            (handle,
             filterDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             op.context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
             &op.context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, op.context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed) ;
    }

    // Get workspace
    if (workSpaceSize > 0) {
      workSpace = op.context.getWorkspace(vl::VLDT_GPU, workSpaceSize) ;
      if (workSpace == NULL) {
        error = op.context.getLastError() ;
        goto done ;
      }
    }
#endif

    // Perform backward convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t filterGrpOffset = filterVolume * numFiltersPerGroup  * g ;
      ptrdiff_t derOutputGrpOffset = (derOutput.getHeight() * derOutput.getWidth() * numFiltersPerGroup) * g ;

      if (derBias) {
        ptrdiff_t derBiasGrpOffset = numFiltersPerGroup * g ;
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
        ptrdiff_t dataGrpOffset = (input.getHeight() * input.getWidth() * derFilter.getDepth()) *  g ;
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
               op.context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
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
        ptrdiff_t dataGrpOffset = (derInput.getHeight() * derInput.getWidth() * filter.getDepth()) *  g ;
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
               op.context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
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
    return op.context.passError(error, __func__) ;
  }

  } ;



