// @file nnbnorm_cudnn.hpp
// @brief bnorm CuDNN-based implementation.
// @author Andrea Vedaldi

/*
 Copyright (C) 2016-17 Andrea Vedaldi.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#include "nnbnorm.hpp"
#include "datacu.hpp"
#include "impl/cudnnhelper.hpp"
#include "impl/copy.hpp"
#include <cassert>

using namespace std ;
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
//                                                             Kernels
// -------------------------------------------------------------------

template<typename T>
__global__ void var_to_std(T * var, unsigned int num, T scale, T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    var[idx] = sqrt(scale * var[idx] + epsilon) ;
  }
}

template<typename T>
__global__ void std_to_var(T * var, T const * std, unsigned int num, T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    var[idx] = std[idx]*std[idx] - epsilon ;
  }
}

template<typename T>
__global__ void inverse(T * ivar, unsigned int num)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    ivar[idx] = ((T)1) / ivar[idx] ;
  }
}

template<typename T>
__global__ void inverse(T * out, T * in, unsigned int num)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    out[idx] = ((T)1) / in[idx] ;
  }
}

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BatchNormForwardWithMomentCudnn
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &output,
                           Tensor const &moment, // can be null
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias)
  {
    assert(output) ;
    assert(input) ;
    assert(multiplier) ;
    assert(bias) ;

    if (op.epsilon < CUDNN_BN_MIN_EPSILON) { return VLE_Unsupported ; }

    typedef typename DataTypeTraits<dataType>::type type ;
    size_t workspaceSize ;
    type * workspace ;

    cudnnTensorDescriptor_t dataDesc, momentDesc ;
    bool dataDescInitialized = false ;
    bool momentDescInitialized = false ;

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN.
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs.
    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     input.getSize(),
                                     input.getDepth(),
                                     input.getWidth(),
                                     input.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     1, input.getDepth(), 1, 1)) ;

    // Allocate workspace.
    workspaceSize = input.getDepth() ;
    workspace = (type*)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;

    // Run CuDNN batch normalization implementation.
    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      type * meanMemory = moment ? (type*)moment.getMemory() : workspace ;
      type * stdMemory = meanMemory + input.getDepth() ;
      type * varMemory = workspace ;

      size_t const blockSize = VL_CUDA_NUM_THREADS ;
      std_to_var<type> <<<divideAndRoundUp(input.getDepth(),blockSize),blockSize>>>
      (varMemory, stdMemory, input.getDepth(), CUDNN_BN_MIN_EPSILON) ;

      CHECK(cudnnBatchNormalizationForwardInference
            (handle,
             CUDNN_BATCHNORM_SPATIAL,
             &alpha,
             &beta,
             dataDesc, input.getMemory(),
             dataDesc, output.getMemory(),
             momentDesc, multiplier.getMemory(), bias.getMemory(),
             meanMemory, varMemory, CUDNN_BN_MIN_EPSILON)) ;
    }

    // Finish.
  done:
    if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    return op.context.passError(error, __func__) ;
  }
} ;

template<DataType dataType>
struct BatchNormForwardCudnn
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &output,
                           Tensor &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias)
  {
    assert(output) ;
    assert(input) ;
    assert(multiplier) ;
    assert(bias) ;

    if (op.epsilon < CUDNN_BN_MIN_EPSILON) { return VLE_Unsupported ; }

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t dataDesc, momentDesc ;
    bool dataDescInitialized = false ;
    bool momentDescInitialized = false ;

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN.
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs.
    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     input.getSize(),
                                     input.getDepth(),
                                     input.getWidth(),
                                     input.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     1, input.getDepth(), 1, 1)) ;

    // Run CuDNN batch normalization implementation.
    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      type * meanMemory = NULL ;
      type * varMemory = NULL ;

      if (moment) {
        meanMemory = (type*)moment.getMemory()  ;
        varMemory = meanMemory + input.getDepth() ;
        vl::impl::operations<vl::VLDT_GPU,type>::fill
        (meanMemory, 2 * input.getDepth() * sizeof(type), 0) ;
      }

      CHECK(cudnnBatchNormalizationForwardTraining
            (handle,
             CUDNN_BATCHNORM_SPATIAL,
             &alpha, &beta,
             dataDesc, input.getMemory(),
             dataDesc, output.getMemory(),
             momentDesc, multiplier.getMemory(), bias.getMemory(),
             0, NULL, NULL,
             op.epsilon,
             meanMemory, varMemory)) ;

      if (varMemory) {
        // CuDNN computes the variance without epsilon, whereas MCN
        // returns the standard deviation after adding epsilon.
        // Also, CuDNN returns the unbiased variance estimate, but it is
        // debatable that this is appropriate.
        //
        // We pick instead the caches, which are closer to the values we compute.
        // Also they do not need to be pre-initialized with zeros.

        size_t const blockSize = VL_CUDA_NUM_THREADS ;
        inverse<type> <<<divideAndRoundUp(input.getDepth(),blockSize),blockSize>>>
        (varMemory, input.getDepth()) ;
      }
    }

    // Finish.
  done:
    if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    return op.context.passError(error, __func__) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct BatchNormBackwardWithMomentCudnn
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &derInput,
                           Tensor &derMultiplier,
                           Tensor &derBias,
                           Tensor const &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias,
                           Tensor const &derOutput)
  {
    assert(derInput) ;
    assert(derMultiplier) ;
    assert(derBias) ;
    assert(moment) ;
    assert(input) ;
    assert(multiplier) ;
    assert(bias) ;
    assert(derOutput) ;

    if (op.epsilon < CUDNN_BN_MIN_EPSILON) { return VLE_Unsupported ; }

    typedef typename DataTypeTraits<dataType>::type type ;
    size_t workspaceSize ;
    type * workspace ;

    cudnnTensorDescriptor_t derOutputDesc, dataDesc, momentDesc ;
    bool derOutputDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool momentDescInitialized = false ;

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = derOutput.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN.
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs.
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     derOutput.getSize(), // sizes
                                     derOutput.getDepth(),
                                     derOutput.getWidth(),
                                     derOutput.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     input.getSize(),
                                     input.getDepth(),
                                     input.getWidth(),
                                     input.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     1, input.getDepth(), 1, 1)) ;


    // Scrarch space to provide moments in CuDNN format.
    workspaceSize = derInput.getDepth() ;
    workspace = (type*)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;

    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      type * meanMemory = (type*)moment.getMemory() ;
      type * stdMemory = meanMemory + input.getDepth() ;
      type * istdMemory = workspace ;

      // The CuDNN manual describes the varMemory output above
      // as inverse variance, but it is the inverse standard deviation instead.
      size_t const blockSize = VL_CUDA_NUM_THREADS ;
      inverse<type> <<<divideAndRoundUp(input.getDepth(),blockSize),blockSize>>>
      (istdMemory, stdMemory, input.getDepth()) ;

      CHECK(cudnnBatchNormalizationBackward
            (handle,
             CUDNN_BATCHNORM_SPATIAL,
             &alpha, &beta, // data
             &alpha, &beta, // params
             dataDesc, input.getMemory(), // input
             derOutputDesc, derOutput.getMemory(), // input
             dataDesc, derInput.getMemory(), // output
             momentDesc, multiplier.getMemory(), // input
             derMultiplier.getMemory(), // output
             derBias.getMemory(), // output
             op.epsilon,
             meanMemory, istdMemory)) ;
    }

    // Finish.
  done:
    if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    return op.context.passError(error, __func__) ;
  }
} ;

template<DataType dataType>
struct BatchNormBackwardCudnn
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &derInput,
                           Tensor &derMultiplier,
                           Tensor &derBias,
                           Tensor &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias,
                           Tensor const &derOutput)
  {
    assert(derInput) ;
    assert(derMultiplier) ;
    assert(derBias) ;
    assert(input) ;
    assert(multiplier) ;
    assert(bias) ;
    assert(derOutput) ;

    if (op.epsilon < CUDNN_BN_MIN_EPSILON) { return VLE_Unsupported ; }

    typedef typename DataTypeTraits<dataType>::type type ;
    size_t workspaceSize ;
    type * workspace ;
    size_t volume ;

    cudnnTensorDescriptor_t derOutputDesc, momentDesc ;
    bool derOutputDescInitialized = false ;
    bool momentDescInitialized = false ;

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = derOutput.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN.
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs.
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     derOutput.getSize(), // sizes
                                     derOutput.getDepth(),
                                     derOutput.getWidth(),
                                     derOutput.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
    momentDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     1, input.getDepth(), 1, 1)) ;

    // Compute moment using CuDNN. Unfortunately CuDNN does not expose
    // the values of the moment in the backward pass, so we need to run
    // the forward code to get them.

    volume = derInput.getNumElements() ;
    workspaceSize = (moment ? 0 : 2 * derInput.getDepth()) + volume ;
    workspace = (type*)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;

    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      type * outMemory = workspace ;
      type * meanMemory = moment ? (type*)moment.getMemory() : workspace + volume ;
      type * varMemory = meanMemory + input.getDepth() ;

      CHECK(cudnnBatchNormalizationForwardTraining
            (handle,
             CUDNN_BATCHNORM_SPATIAL,
             &alpha, &beta,
             derOutputDesc, input.getMemory(),
             derOutputDesc, outMemory, // will be discarded
             momentDesc, multiplier.getMemory(), bias.getMemory(),
             1.0, // cumulative factor for moment
             NULL, NULL,
             op.epsilon,
             meanMemory, varMemory)) ;

      CHECK(cudnnBatchNormalizationBackward
            (handle,
             CUDNN_BATCHNORM_SPATIAL,
             &alpha, &beta, // data
             &alpha, &beta, // params
             derOutputDesc, input.getMemory(), // input
             derOutputDesc, derOutput.getMemory(), // input
             derOutputDesc, derInput.getMemory(), // output
             momentDesc, multiplier.getMemory(), // input
             derMultiplier.getMemory(), // output
             derBias.getMemory(), // output
             op.epsilon,
             meanMemory, varMemory)) ;

      // The CuDNN manual describes the varMemory output above
      // as inverse variance, but it is the inverse standard deviation instead.
      size_t const blockSize = VL_CUDA_NUM_THREADS ;
      inverse<type> <<<divideAndRoundUp(input.getDepth(),blockSize),blockSize>>>
      (varMemory, input.getDepth()) ;
    }

    // Finish.
  done:
    if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    return op.context.passError(error, __func__) ;
  }
} ;

