// @file nnconv_blas.cu
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnconv_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnconv_cudnn.hpp"
#include "../datacu.hpp"
#include <assert.h>

using namespace vl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
  error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
     STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
  goto done ; \
} }

/* ---------------------------------------------------------------- */
/*                                             nnconv_forward_cudnn */
/* ---------------------------------------------------------------- */

template<> vl::Error
vl::impl::nnconv_forward_cudnn<float>(Context& context,
                                      Tensor output, double outputMult,
                                      Tensor data, double dataMult,
                                      Tensor filters,
                                      Tensor biases,
                                      int strideY, int strideX,
                                      int padTop, int padBottom,
                                      int padLeft, int padRight)
{
  assert(output) ;
  assert(data) ;
  assert(filters) ;

  cudnnTensorDescriptor_t outputDesc, biasesDesc, dataDesc ;
  cudnnFilterDescriptor_t filtersDesc ;
  cudnnConvolutionDescriptor_t convDesc ;
  cudnnConvolutionFwdAlgo_t algo ;
  bool outputDescInitialized = false ;
  bool biasesDescInitialized = false ;
  bool dataDescInitialized = false ;
  bool filtersDescInitialized = false ;
  bool convDescInitialized = false ;

  void* workSpace = NULL ;
  size_t workSpaceSize ;

  int numGroups = data.getDepth() / filters.getDepth() ;
  int numFiltersPerGroup = filters.getSize() / numGroups ;

  if (padLeft != padRight) return vl::vlErrorUnsupported ;
  if (padTop != padBottom) return vl::vlErrorUnsupported ;
  if (filters.getHeight() > data.getHeight()) return vl::vlErrorUnsupported ;
  if (filters.getWidth() > data.getWidth()) return vl::vlErrorUnsupported ;

  cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
  vl::Error error = vl::vlSuccess ;
  cudnnHandle_t handle ;

  // Get CuDNN
  CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

  // Get tensor descripotrs
  CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
  outputDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptorEx(outputDesc,
                                     CUDNN_DATA_FLOAT,
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
                                     CUDNN_DATA_FLOAT,
                                     data.getSize(),
                                     data.getDepth() / numGroups,
                                     data.getWidth(),
                                     data.getHeight(),
                                     data.getHeight()*data.getWidth()*data.getDepth(), //strides
                                     data.getHeight()*data.getWidth(),
                                     data.getHeight(),
                                     1)) ;

  CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
  filtersDescInitialized = true ;
  CHECK(cudnnSetFilter4dDescriptor(filtersDesc,
                                   CUDNN_DATA_FLOAT,
                                   numFiltersPerGroup,
                                   filters.getDepth(),
                                   filters.getWidth(),
                                   filters.getHeight())) ;

  if (biases) {
    CHECK(cudnnCreateTensorDescriptor(&biasesDesc)) ;
    biasesDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(biasesDesc,
                                     CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT,
                                     1,
                                     biases.getNumElements() / numGroups,
                                     1,
                                     1)) ;
  }

  // Get convolution descriptor
  CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
  convDescInitialized = true ;
  CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                        padLeft, padTop,
                                        strideX, strideY,
                                        1,1, // upscale
                                        CUDNN_CROSS_CORRELATION)) ;
  // Sanity check
#if 1
  {
    int n, c, h, w ;
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          dataDesc,
                                          filtersDesc,
                                          &n, &c, &w, &h) ;
    bool sane =
    output.getSize() == n &&
    numFiltersPerGroup == c &&
    output.getWidth() == w &&
    output.getHeight() == h ;
    assert(sane) ;
  }
#endif

  // Get convolution algorithm
  CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                            dataDesc,
                                            filtersDesc,
                                            convDesc,
                                            outputDesc,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            0,
                                            &algo)) ;

  // Get workspace size
  CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                dataDesc,
                                                filtersDesc,
                                                convDesc,
                                                outputDesc,
                                                algo,
                                                &workSpaceSize)) ;

  // Get workspace
  if (workSpaceSize > 0) {
    workSpace = context.getWorkspace(vl::GPU, workSpaceSize) ;
    if (workSpace == NULL) {
      error = context.getLastError() ;
      goto done ;
    }
  }

  // Perform convolution for each filter group
  for (int g = 0  ; g < numGroups ; ++g) {
    ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * filters.getDepth()) *  g ;
    ptrdiff_t filtersGrpOffset = (filters.getHeight() * filters.getWidth() * filters.getDepth()) * numFiltersPerGroup * g ;
    ptrdiff_t outputGrpOffset = (output.getHeight() * output.getWidth() * numFiltersPerGroup) * g ;
    ptrdiff_t biasesGrpOffset = numFiltersPerGroup * g ;

    float alpha = dataMult ;
    float beta = outputMult ;
    CHECK(cudnnConvolutionForward(handle,
                                  &alpha,
                                  dataDesc, data.getMemory() + dataGrpOffset,
                                  filtersDesc, filters.getMemory() + filtersGrpOffset,
                                  convDesc,
                                  algo,
                                  workSpace, workSpaceSize,
                                  &beta,
                                  outputDesc, output.getMemory() + outputGrpOffset)) ;

    if (biases) {
      float alpha = 1.0f ;
      float beta = 1.0f ;
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_SAME_C,
                           &alpha,
                           biasesDesc, biases.getMemory() + biasesGrpOffset,
                           &beta,
                           outputDesc, output.getMemory() + outputGrpOffset)) ;
    }
  }

  /* cleanup */
done:
  if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
  if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
  if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
  if (biasesDescInitialized) { cudnnDestroyTensorDescriptor(biasesDesc) ; }
  if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
  return context.passError(error, "nnconv_forward_cudnn: ") ;
}

/* ---------------------------------------------------------------- */
/*                                            nnconv_backward_cudnn */
/* ---------------------------------------------------------------- */

template<> vl::Error
vl::impl::nnconv_backward_cudnn<float>(Context& context,
                                       Tensor derData,
                                       Tensor derFilters,
                                       Tensor derBiases,
                                       Tensor data,
                                       Tensor filters,
                                       Tensor derOutput,
                                       int strideY, int strideX,
                                       int padTop, int padBottom,
                                       int padLeft, int padRight)
{


  /* no derDataDesc needed as same as dataDesc */
  cudnnTensorDescriptor_t dataDesc, derBiasesDesc, derOutputDesc ;
  cudnnFilterDescriptor_t filtersDesc ;
  cudnnConvolutionDescriptor_t convDesc ;
  bool dataDescInitialized = false ;
  bool derBiasesDescInitialized = false ;
  bool derOutputDescInitialized = false ;
  bool filtersDescInitialized = false ;
  bool convDescInitialized = false ;

  ptrdiff_t numGroups = 1 ;
  ptrdiff_t numFiltersPerGroup = 0 ;
  ptrdiff_t filtersVolume = 0 ;

  if (padLeft != padRight) return vl::vlErrorUnsupported ;
  if (padTop != padBottom) return vl::vlErrorUnsupported ;

  cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
  vl::Error error = vl::vlSuccess ;
  cudnnHandle_t handle ;

  // Get CuDNN
  CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

  // Get tensor descripotrs

  // for derivative w.r.t. data
  if (derData) {
    assert(filters) ;
    numGroups = derData.getDepth() / filters.getDepth() ;
    numFiltersPerGroup = filters.getSize() / numGroups ;
    filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       CUDNN_DATA_FLOAT,
                                       derData.getSize(),
                                       derData.getDepth() / numGroups,
                                       derData.getWidth(),
                                       derData.getHeight(),
                                       derData.getHeight()*derData.getWidth()*derData.getDepth(), //strides
                                       derData.getHeight()*derData.getWidth(),
                                       derData.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
    filtersDescInitialized = true ;
    CHECK(cudnnSetFilter4dDescriptor(filtersDesc,
                                     CUDNN_DATA_FLOAT,
                                     numFiltersPerGroup,
                                     filters.getDepth(),
                                     filters.getWidth(),
                                     filters.getHeight())) ;
  } else if (derFilters) {
    assert(data) ;
    numGroups = data.getDepth() / derFilters.getDepth() ;
    numFiltersPerGroup = derFilters.getSize() / numGroups ;
    filtersVolume = derFilters.getHeight() * derFilters.getWidth() * derFilters.getDepth() ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       CUDNN_DATA_FLOAT,
                                       data.getSize(),
                                       data.getDepth() / numGroups,
                                       data.getWidth(),
                                       data.getHeight(),
                                       data.getHeight()*data.getWidth()*data.getDepth(), //strides
                                       data.getHeight()*data.getWidth(),
                                       data.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
    filtersDescInitialized = true ;
    CHECK(cudnnSetFilter4dDescriptor(filtersDesc,
                                     CUDNN_DATA_FLOAT,
                                     numFiltersPerGroup,
                                     derFilters.getDepth(),
                                     derFilters.getWidth(),
                                     derFilters.getHeight())) ;
  }

  CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
  convDescInitialized = true ;
  CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                        padLeft, padTop,
                                        strideX, strideY,
                                        1,1, // upscale
                                        CUDNN_CROSS_CORRELATION)) ;

  // Must have derOutput for all derivatives
  assert(derOutput) ;
  CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
  derOutputDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptorEx(derOutputDesc,
                                     CUDNN_DATA_FLOAT,
                                     derOutput.getSize(), // sizes
                                     numFiltersPerGroup,
                                     derOutput.getWidth(),
                                     derOutput.getHeight(),
                                     derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth(), //strides
                                     derOutput.getHeight()*derOutput.getWidth(),
                                     derOutput.getHeight(),
                                     1)) ;

  // for derivatives w.r.t. bias
  if (derBiases) {
    CHECK(cudnnCreateTensorDescriptor(&derBiasesDesc)) ;
    derBiasesDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(derBiasesDesc,
                                     CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT,
                                     1,
                                     derBiases.getNumElements() / numGroups,
                                     1,
                                     1)) ;
  }
  

  // Perform backward convolution for each filter group
  for (int g = 0  ; g < numGroups ; ++g) {
    ptrdiff_t filtersGrpOffset = filtersVolume * numFiltersPerGroup  * g ;
    ptrdiff_t derOutputGrpOffset = (derOutput.getHeight() * derOutput.getWidth() * numFiltersPerGroup) * g ;

    if (derBiases) {
      ptrdiff_t derBiasesGrpOffset = numFiltersPerGroup * g ;
      float alpha = 1 ;
      float beta = 0 ;
      CHECK(cudnnConvolutionBackwardBias
      (handle,
       &alpha,
       derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
       &beta,
       derBiasesDesc, derBiases.getMemory() + derBiasesGrpOffset)) ;
    }
    if (derFilters) {
      ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * derFilters.getDepth()) *  g ;
      float alpha = 1 ;
      float beta = 0 ;
      CHECK(cudnnConvolutionBackwardFilter
      (handle,
       &alpha,
       dataDesc, data.getMemory() + dataGrpOffset,
       derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
       convDesc,
       &beta,
       filtersDesc, derFilters.getMemory() + filtersGrpOffset)) ;
    }
    if (derData) {
      ptrdiff_t dataGrpOffset = (derData.getHeight() * derData.getWidth() * filters.getDepth()) *  g ;
      float alpha = 1 ;
      float beta = 0 ;
      CHECK(cudnnConvolutionBackwardData
      (handle,
       &alpha,
       filtersDesc, filters.getMemory() + filtersGrpOffset,
       derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
       convDesc,
       &beta,
       dataDesc, derData.getMemory() + dataGrpOffset)) ;
    }
  }

done:
  if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
  if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
  if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
  if (derBiasesDescInitialized) { cudnnDestroyTensorDescriptor(derBiasesDesc) ; }
  if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
  return context.passError(error, "nnconv_backward_cudnn: ") ;
}

