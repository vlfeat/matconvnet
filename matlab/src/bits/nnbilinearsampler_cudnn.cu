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
#include <assert.h>
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
  vl::impl::nnbilinearsampler_cudnn<dataType>::forward(Context& op.context,
                                                       Tensor output,
                                                       Tensor data,
                                                       Tensor grid)
  {
    return vl::VLE_Unsupported ;
  }

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnbilinearsampler_cudnn<dataType>::backward(Context& op.context,
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
error = op.context.setError(op.context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__))) ; \
goto done ; \
} }

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BilinearSamplerForwardCudnn
{
  vl::ErrorCode operator()(BilinearSampler &op,
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
    int inCardinality = input.getSize();
    int inDepth = input.getDepth();
    int inHeight = input.getHeight();
    int inWidth = input.getWidth();

    int outCardinality = output.getSize();
    int outDepth = output.getDepth();
    int outWidth = output.getWidth();
    int outHeight = output.getHeight();

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // get number of transforms/image == groupSize:
    int groupSize = outCardinality / inCardinality ;
    int dimOut[4] = { 1, outDepth, outWidth, outHeight } ; // one-image

    // Get CuDNN
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

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
                                       1, inDepth, inWidth, inHeight, // sizes: n,c,w,h
                                       inHeight * inWidth * inDepth, //strides
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
      const ptrdiff_t dataOffset = inHeight * inWidth * inDepth ;
      const ptrdiff_t gridOffset = 2 * outWidth * outHeight ;
      const ptrdiff_t outOffset = outHeight * outWidth * outDepth ;
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
    return op.context.passError(error, __func__) ;
  }
};

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BilinearSamplerBackwardCudnn
{
  vl::ErrorCode operator()
  (BilinearSampler &op,
   Tensor &derInput,
   Tensor &derGrid,
   Tensor const &input,
   Tensor const &grid,
   Tensor const &derOutput)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derInputDataDesc needed as same as dataDesc <-- nice! */
    cudnnTensorDescriptor_t dataDesc, derOutputDesc ;
    cudnnSpatialTransformerDescriptor_t samplerDesc ;
    bool dataDescInitialized = false ;
    bool derOutputDescInitialized = false ;
    bool samplerDescInitialized = false ;

    // get the sizes:
    int inCardinality = input.getSize();
    int inDepth = input.getDepth();
    int inHeight = input.getHeight();
    int inWidth = input.getWidth();

    int outCardinality = derOutput.getSize();
    int outDepth = derOutput.getDepth();
    int outWidth = derOutput.getWidth();
    int outHeight = derOutput.getHeight();

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = derOutput.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // get number of transforms/image == groupSize:
    int groupSize = outCardinality / inCardinality;
    int dimOut[4] = { 1, outDepth, outWidth, outHeight };

    // Get CuDNN
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;


    // Get tensor descriptors:
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
                                       1, inDepth, inWidth, inHeight, // sizes: n,c,w,h
                                       inHeight * inWidth * inDepth, //strides
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
      const ptrdiff_t dataOffset = inHeight * inWidth * inDepth ;
      const ptrdiff_t gridOffset = 2 * outWidth * outHeight ;
      const ptrdiff_t outOffset = outHeight * outWidth * outDepth ;
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
    return op.context.passError(error, __func__) ;
  }
} ;

#endif // CUDNN >= v5.0
