// @file nnbilinearsampler_cudnn.cu
// @brief BilinearSampler CuDNN-based implementation.
// @author Ankush Gupta, Andrea Vedaldi

/*
Copyright (C) 2016-17 Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbilinearsampler.hpp"
#include "datacu.hpp"
#include "impl/cudnnhelper.hpp"
#include <cassert>
#include <algorithm>

using namespace std ;
using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

#if CUDNN_VERSION < 5000
#warning "bilinearsampler_cudnn.cu will be disabled as it requires CUDNN v5 or higher."

namespace vl { namespace impl {
  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnbilinearsampler_cudnn<dataType>::forward(Context& op.getContext(),
                                                       Tensor output,
                                                       Tensor data,
                                                       Tensor grid)
  {
    return vl::VLE_Unsupported ;
  }

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnbilinearsampler_cudnn<dataType>::backward(Context& op.getContext(),
                                                        Tensor derInputData,
                                                        Tensor derGrid,
                                                        Tensor data,
                                                        Tensor grid,
                                                        Tensor derOutput)
  {
    return vl::VLE_Unsupported ;
  }
}}
#else // CUDNN_VERSION

// check if the descriptors, etc. were successfully created:
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
struct BilinearSamplerForwardCudnn
{
  vl::ErrorCode operator()(BilinearSampler const &op,
                           Tensor &output,
                           Tensor const &input,
                           Tensor const &grid)
  {
    assert(output) ;
    assert(input) ;
    assert(grid) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, dataDesc ;
    cudnnSpatialTransformerDescriptor_t samplerDesc ;
    bool outputDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool samplerDescInitialized = false ;

    // get the sizes:
    int inCardinality = (int)input.getCardinality();
    int inNumChannels = (int)input.getNumChannels();
    int inHeight = (int)input.getHeight();
    int inWidth = (int)input.getWidth();

    int outCardinality = (int)output.getCardinality();
    int outDepth = (int)output.getNumChannels();
    int outWidth = (int)output.getWidth();
    int outHeight = (int)output.getHeight();

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // get number of transforms/image == groupSize:
    int groupSize = (int)(outCardinality / inCardinality) ;
    int dimOut[4] = { 1, (int)outDepth, (int)outWidth, (int)outHeight } ; // one-image

    // Get CuDNN
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors:
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(outputDesc,
                                       cudnnDataType,
                                       1, outDepth, outWidth, outHeight, // sizes: n,c,w,h
                                       outHeight * outWidth * outDepth, //strides
                                       outHeight * outWidth,
                                       outHeight,
                                       1)) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       cudnnDataType,
                                       1, inNumChannels, inWidth, inHeight, // sizes: n,c,w,h
                                       inHeight * inWidth * inNumChannels, //strides
                                       inHeight * inWidth,
                                       inHeight,
                                       1)) ;

    // Get bilinear-sampler descriptor:
    CHECK(cudnnCreateSpatialTransformerDescriptor(&samplerDesc)) ;
    samplerDescInitialized = true ;
    CHECK(cudnnSetSpatialTransformerNdDescriptor(samplerDesc,
                                                 CUDNN_SAMPLER_BILINEAR,
                                                 cudnnDataType,
                                                 4,
                                                 dimOut)) ;

    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      const Int dataOffset = inHeight * inWidth * inNumChannels ;
      const Int gridOffset = 2 * outWidth * outHeight ;
      const Int outOffset = outHeight * outWidth * outDepth ;
      type const* data_ptr = (type const*) input.getMemory() ;
      type const* grid_ptr = (type const*) grid.getMemory() ;
      type * out_ptr = (type *) output.getMemory() ;

      for (int im=0; im < inCardinality; im++) {
        for (int ig=0; ig < groupSize; ig++) {
          cudnnSpatialTfSamplerForward(handle,
                                       samplerDesc,
                                       &alpha,
                                       dataDesc, data_ptr,
                                       grid_ptr,
                                       &beta,
                                       outputDesc, out_ptr) ;
          grid_ptr += gridOffset ;
          out_ptr += outOffset ;
        }
        data_ptr += dataOffset ;
      }
    }

  done:
    if (samplerDescInitialized) { cudnnDestroySpatialTransformerDescriptor(samplerDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return op.getContext().passError(error, __func__) ;
  }
};

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BilinearSamplerBackwardCudnn
{
  vl::ErrorCode operator()
  (BilinearSampler const &op,
   Tensor &derInput,
   Tensor &derGrid,
   Tensor const &input,
   Tensor const &grid,
   Tensor const &derOutput)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    // No derInputDataDesc needed as same as dataDesc.
    cudnnTensorDescriptor_t dataDesc, derOutputDesc ;
    cudnnSpatialTransformerDescriptor_t samplerDesc ;
    bool dataDescInitialized = false ;
    bool derOutputDescInitialized = false ;
    bool samplerDescInitialized = false ;

    int inCardinality = (int)derInput.getCardinality();
    int inNumChannels = (int)derInput.getNumChannels();
    int inHeight = (int)derInput.getHeight();
    int inWidth = (int)derInput.getWidth();

    int outCardinality = (int)derOutput.getCardinality();
    int outDepth = (int)derOutput.getNumChannels();
    int outWidth = (int)derOutput.getWidth();
    int outHeight = (int)derOutput.getHeight();

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = derOutput.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get number of transforms/image == groupSize.
    int groupSize = (int)(outCardinality / inCardinality) ;
    int dimOut[4] = { 1, (int)outDepth, (int)outWidth, (int)outHeight };

    // Get CuDNN.
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors.
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(derOutputDesc,
                                       cudnnDataType,
                                       1, outDepth, outWidth, outHeight, // sizes: n,c,w,h
                                       outHeight * outWidth * outDepth, //strides
                                       outHeight * outWidth,
                                       outHeight,
                                       1)) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       cudnnDataType,
                                       1, inNumChannels, inWidth, inHeight, // sizes: n,c,w,h
                                       inHeight * inWidth * inNumChannels, //strides
                                       inHeight * inWidth,
                                       inHeight,
                                       1)) ;

    // Get bilinear-sampler descriptor:
    CHECK(cudnnCreateSpatialTransformerDescriptor(&samplerDesc)) ;
    samplerDescInitialized = true ;
    CHECK(cudnnSetSpatialTransformerNdDescriptor(samplerDesc,
                                                 CUDNN_SAMPLER_BILINEAR,
                                                 cudnnDataType,
                                                 4,
                                                 dimOut));
    /* do the work */
    {
      type alpha = 1.0f ;
      type dataBeta = 1.0f ; // assuming that the derInputData has been initialized to zero
      type gridBeta = 0.0f ;
      int const dataOffset = inHeight * inWidth * inNumChannels ;
      int const gridOffset = 2 * outWidth * outHeight ;
      int const outOffset = outHeight * outWidth * outDepth ;
      type const* data_ptr = (type const*) input.getMemory() ;
      type * derInputData_ptr = (type *) derInput.getMemory() ;
      type const* grid_ptr = (type const*) grid.getMemory() ;
      type * derGrid_ptr = (type *) derGrid.getMemory() ;
      type * derOut_ptr = (type *) derOutput.getMemory() ;

      for (int im=0; im < inCardinality; im++) {
        for (int ig=0; ig < groupSize; ig++) {
        cudnnSpatialTfSamplerBackward(handle,
                                      samplerDesc,
                                      &alpha,
                                      dataDesc, data_ptr,
                                      &dataBeta,
                                      dataDesc, derInputData_ptr,
                                      &alpha,
                                      derOutputDesc, derOut_ptr,
                                      grid_ptr,
                                      &gridBeta,
                                      derGrid_ptr) ;
          grid_ptr += gridOffset ;
          derGrid_ptr += gridOffset ;
          derOut_ptr += outOffset ;
        }
        data_ptr += dataOffset ;
        derInputData_ptr += dataOffset ;
      }
    }

  /* cleanup */
  done:
    if (samplerDescInitialized) { cudnnDestroySpatialTransformerDescriptor(samplerDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    return op.getContext().passError(error, __func__) ;
  }
} ;

#endif // CUDNN >= v5.0
