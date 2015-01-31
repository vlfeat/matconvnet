//
//  recipies_cudnn.cu
//  matconv
//
//  Created by Andrea Vedaldi on 30/01/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "nnconv_cudnn.hpp"
#include <assert.h>
#include <iostream.h>

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnconv_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "../datacu.hpp"

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                             nnconv_forward_cudnn */
/* ---------------------------------------------------------------- */

template<> int
vl::impl::nnconv_forward_cudnn<float>(Context& context,
                                      Tensor output,
                                      Tensor data,
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
  cudnnFilterDescriptor_t filtersDesc;
  cudnnConvolutionDescriptor_t convDesc ;
  cudnnConvolutionFwdAlgo_t algo;
  void* workSpace = NULL ;
  size_t workSpaceSize ;

  int numGroups = data.getDepth() / filters.getDepth() ;
  int numFiltersPerGroup = filters.getSize() / numGroups ;

  //std::cout<<"numGroups"<<numGroups<<std::endl;
  //std::cout<<"numFiltersPerGroup"<<numFiltersPerGroup<<std::endl;

  if (padLeft != padRight) return 1 ;
  if (padTop != padBottom) return 1 ;

  cudnnHandle_t handle;
  context.getCudaHelper().getCuDNNHandle(&handle) ;

  cudnnCreateTensorDescriptor(&outputDesc) ;
  cudnnSetTensor4dDescriptorEx(outputDesc,
                               CUDNN_DATA_FLOAT,
                               output.getSize(), // sizes
                               numFiltersPerGroup,
                               output.getWidth(),
                               output.getHeight(),
                               output.getHeight()*output.getWidth()*output.getDepth(), //strides
                               output.getHeight()*output.getWidth(),
                               output.getHeight(),
                               1) ;

  cudnnCreateTensorDescriptor(&dataDesc) ;
  cudnnSetTensor4dDescriptorEx(dataDesc,
                               CUDNN_DATA_FLOAT,
                               data.getSize(),
                               data.getDepth() / numGroups,
                               data.getWidth(),
                               data.getHeight(),
                               data.getHeight()*data.getWidth()*data.getDepth(), //strides
                               data.getHeight()*data.getWidth(),
                               data.getHeight(),
                               1) ;

  cudnnCreateFilterDescriptor(&filtersDesc) ;
  cudnnSetFilter4dDescriptor(filtersDesc,
                             CUDNN_DATA_FLOAT,
                             numFiltersPerGroup,
                             filters.getDepth(),
                             filters.getWidth(),
                             filters.getHeight()) ;

  if (biases) {
    cudnnCreateTensorDescriptor(&biasesDesc) ;
    cudnnSetTensor4dDescriptor(biasesDesc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               1,
                               biases.getNumElements() / numGroups,
                               1,
                               1) ;
  }

  cudnnCreateConvolutionDescriptor(&convDesc) ;
  cudnnSetConvolution2dDescriptor(convDesc,
                                  padLeft, padTop,
                                  strideX, strideY,
                                  1,1, // upscale
                                  CUDNN_CROSS_CORRELATION) ;
  /* sanity check */
  {
    int n, c, h, w ;
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          dataDesc,
                                          filtersDesc,
                                          &n, &c, &w, &h) ;
    bool ok =
    output.getSize() == n &&
    numFiltersPerGroup == c &&
    output.getWidth() == w &&
    output.getHeight() == h ;

    if (!ok) {

      std::cout<<"eeeeeee: "<<output.getWidth()<<" "<<w<<std::endl ;
      std::cout<<"qqqqqqq: "<<output.getHeight()<<" "<<h<<std::endl ;
      /*
       assert(output.getSize() == n &&
       output.getDepth() == c &&
       output.getWidth() == w &&
       output.getHeight() == h) ;
       */
    }
  }

  /* pick convolution algorithm */
  cudnnGetConvolutionForwardAlgorithm(handle,
                                      dataDesc,
                                      filtersDesc,
                                      convDesc,
                                      outputDesc,
                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                      0,
                                      &algo) ;

  /* get workspace */
  cudnnGetConvolutionForwardWorkspaceSize(handle,
                                          dataDesc,
                                          filtersDesc,
                                          convDesc,
                                          outputDesc,
                                          algo,
                                          &workSpaceSize) ;
  workSpace = context.getWorkspace(vl::GPU, workSpaceSize) ;

  /* peform convolution */
  for (int g = 0  ; g < numGroups ; ++g) {
    ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * filters.getDepth()) *  g ;

    ptrdiff_t filtersGrpOffset = (filters.getHeight() * filters.getWidth() * filters.getDepth()) * numFiltersPerGroup * g ;
    ptrdiff_t outputGrpOffset = (output.getHeight() * output.getWidth() * numFiltersPerGroup) * g ;
    ptrdiff_t biasesGrpOffset = numFiltersPerGroup * g ;

    float alpha = 1.0f ;
    float beta = 0.0f ;
    cudnnConvolutionForward(handle,
                            &alpha,
                            dataDesc, data.getMemory() + dataGrpOffset,
                            filtersDesc, filters.getMemory() + filtersGrpOffset,
                            convDesc,
                            algo,
                            workSpace, workSpaceSize,
                            &beta,
                            outputDesc, output.getMemory() + outputGrpOffset) ;

    if (biases) {
      float alpha = 1.0f ;
      float beta = 1.0f ;
      cudnnAddTensor(handle,
                     CUDNN_ADD_SAME_C,
                     &alpha,
                     biasesDesc, biases.getMemory() + biasesGrpOffset,
                     &beta,
                     outputDesc, output.getMemory() + outputGrpOffset);
    }
  }

  /* cleanup */
  cudnnDestroyConvolutionDescriptor(convDesc) ;
  cudnnDestroyFilterDescriptor(filtersDesc) ;
  cudnnDestroyTensorDescriptor(dataDesc) ;
  if (biases) { cudnnDestroyTensorDescriptor(biasesDesc) ; }
  cudnnDestroyTensorDescriptor(outputDesc) ;
  return 0 ;
}

/* ---------------------------------------------------------------- */
/*                                            nnconv_backward_cudnn */
/* ---------------------------------------------------------------- */

template<> int
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
  assert(data) ;
  assert(filters) ;
  assert(derOutput) ;

  /* no derDataDesc needed as same as dataDesc */
  cudnnTensorDescriptor_t dataDesc, derBiasesDesc, derOutputDesc ;
  cudnnFilterDescriptor_t filtersDesc ;
  cudnnConvolutionDescriptor_t convDesc ;

  if (padLeft != padRight) return 1 ;
  if (padTop != padBottom) return 1 ;

  int numGroups = data.getDepth() / filters.getDepth() ;
  int numFiltersPerGroup = filters.getSize() / numGroups ;

  cudnnHandle_t handle;
  context.getCudaHelper().getCuDNNHandle(&handle) ;

  cudnnCreateTensorDescriptor(&dataDesc) ;
  cudnnSetTensor4dDescriptorEx(dataDesc,
                               CUDNN_DATA_FLOAT,
                               data.getSize(),
                               data.getDepth() / numGroups,
                               data.getWidth(),
                               data.getHeight(),
                               data.getHeight()*data.getWidth()*data.getDepth(), //strides
                               data.getHeight()*data.getWidth(),
                               data.getHeight(),
                               1) ;

  if (derBiases) {
    cudnnCreateTensorDescriptor(&derBiasesDesc) ;
    cudnnSetTensor4dDescriptor(derBiasesDesc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               1,
                               derBiases.getNumElements() / numGroups,
                               1,
                               1) ;
  }

  if (derOutput) {
    cudnnCreateTensorDescriptor(&derOutputDesc) ;
    cudnnSetTensor4dDescriptorEx(derOutputDesc,
                                 CUDNN_DATA_FLOAT,
                                 derOutput.getSize(), // sizes
                                 numFiltersPerGroup,
                                 derOutput.getWidth(),
                                 derOutput.getHeight(),
                                 derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth(), //strides
                                 derOutput.getHeight()*derOutput.getWidth(),
                                 derOutput.getHeight(),
                                 1) ;
  }

  cudnnCreateFilterDescriptor(&filtersDesc) ;
  cudnnSetFilter4dDescriptor(filtersDesc,
                             CUDNN_DATA_FLOAT,
                             numFiltersPerGroup,
                             filters.getDepth(),
                             filters.getWidth(),
                             filters.getHeight()) ;

  cudnnCreateConvolutionDescriptor(&convDesc) ;
  cudnnSetConvolution2dDescriptor(convDesc,
                                  padLeft, padTop,
                                  strideX, strideY,
                                  1,1, // upscale
                                  CUDNN_CROSS_CORRELATION) ;

  for (int g = 0  ; g < numGroups ; ++g) {

    ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * filters.getDepth()) *  g ;
    ptrdiff_t filtersGrpOffset = (filters.getHeight() * filters.getWidth() * filters.getDepth()) * numFiltersPerGroup  * g ;
    ptrdiff_t derOutputGrpOffset = (derOutput.getHeight() * derOutput.getWidth() * numFiltersPerGroup) * g ;
    ptrdiff_t derBiasesGrpOffset = numFiltersPerGroup * g ;

    if (derBiases) {
      float alpha = 1 ;
      float beta = 0 ;
      cudnnConvolutionBackwardBias
      (handle,
       &alpha,
       derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
       &beta,
       derBiasesDesc, derBiases.getMemory() + derBiasesGrpOffset) ;
    }
    if (derFilters) {
      float alpha = 1 ;
      float beta = 0 ;
      cudnnConvolutionBackwardFilter
      (handle,
       &alpha,
       dataDesc, data.getMemory() + dataGrpOffset,
       derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
       convDesc,
       &beta,
       filtersDesc, derFilters.getMemory() + filtersGrpOffset) ;
    }
    if (derData) {
      float alpha = 1 ;
      float beta = 0 ;
      cudnnConvolutionBackwardData
      (handle,
       &alpha,
       filtersDesc, filters.getMemory() + filtersGrpOffset,
       derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
       convDesc,
       &beta,
       dataDesc, derData.getMemory() + dataGrpOffset) ;
    }
  }

  cudnnDestroyConvolutionDescriptor(convDesc) ;
  cudnnDestroyFilterDescriptor(filtersDesc) ;
  if (derOutput) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
  if (derBiases) { cudnnDestroyTensorDescriptor(derBiasesDesc) ; }
  cudnnDestroyTensorDescriptor(dataDesc) ;
  return 0 ;
}

