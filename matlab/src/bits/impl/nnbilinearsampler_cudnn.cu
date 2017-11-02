// @file nnbilinearsampler_cudnn.cu
// @brief BilinearSampler CuDNN-based implementation.
// @author Ankush Gupta, Andrea Vedaldi

/*
Copyright (C) 2016-17 Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../datacu.hpp"
#include "cudnnhelper.hpp"
#include <cassert>
#include <algorithm>

#if CUDNN_VERSION < 5000
#warning "bilinearsampler_cudnn.cu will be disabled as it requires CUDNN v5 or higher."

namespace vl { namespace impl {
  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnbilinearsampler_cudnn<dataType>::forward(Context& op.getContext(),
                                                       Tensor const &output,
                                                       Tensor const &data,
                                                       Tensor const &grid)
  {
    return vl::VLE_Unsupported ;
  }

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnbilinearsampler_cudnn<dataType>::backward(Context& op.getContext(),
                                                        Tensor &derInputData,
                                                        Tensor &derGrid,
                                                        Tensor const &data,
                                                        Tensor const &grid,
                                                        Tensor const &derOutput)
  {
    return vl::VLE_Unsupported ;
  }
}}
#else

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
    static const std::string signature = std::string("BilinearSamplerForward[CuDNN")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    assert(output) ;
    assert(input) ;
    assert(grid) ;

    typedef typename DataTypeTraits<dataType>::type type ;
    Int inCardinality = input.getCardinality();
    Int inNumChannels = input.getNumChannels();
    Int inHeight = input.getHeight();
    Int inWidth = input.getWidth();
    Int outCardinality = output.getCardinality();
    Int outNumChannels = output.getNumChannels();
    Int outWidth = output.getWidth();
    Int outHeight = output.getHeight();
    Int groupSize = outCardinality / inCardinality ;

    // Get CuDNN
    cudnnHandle_t handle ;
    CKCUDNN(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors.
    CudnnTensorDescriptor outputDesc ;
    CKCUDNN(outputDesc.init(dataType,{outHeight,outWidth,outNumChannels,1})) ;

    CudnnTensorDescriptor dataDesc ;
    CKCUDNN(dataDesc.init(dataType,{inHeight,inWidth,inNumChannels,1})) ;

    // Get bilinear-sampler descriptor.
    cudnnSpatialTransformerDescriptor_t samplerDesc ;
    CKCUDNN(cudnnCreateSpatialTransformerDescriptor(&samplerDesc)) ;
    auto transformerDescDeleter = defer([&]{cudnnDestroySpatialTransformerDescriptor(samplerDesc);}) ;
    int dimOut[4] = {1, (int)outNumChannels, (int)outWidth, (int)outHeight} ;
    CKCUDNN(cudnnSetSpatialTransformerNdDescriptor(samplerDesc,
                                                   CUDNN_SAMPLER_BILINEAR,
                                                   DataTypeToCudnn<dataType>::dataType,
                                                   4,
                                                   dimOut)) ;

    type alpha = 1.0f ;
    type beta = 0.0f ;
    type const* data_ptr = (type const*) input.getMemory() ;
    type const* grid_ptr = (type const*) grid.getMemory() ;
    type * out_ptr = (type *) output.getMemory() ;

    // Todo: make faster, we should not need soo many loops.
    Int dataOffset = inHeight * inWidth * inNumChannels ;
    Int gridOffset = 2 * outWidth * outHeight ;
    Int outOffset = outHeight * outWidth * outNumChannels ;
    for (Int im=0; im < inCardinality; im++) {
      for (Int ig=0; ig < groupSize; ig++) {
        CKCUDNN(cudnnSpatialTfSamplerForward(handle,
                                             samplerDesc,
                                             &alpha,
                                             dataDesc.get(), data_ptr,
                                             grid_ptr,
                                             &beta,
                                             outputDesc.get(), out_ptr)) ;
        grid_ptr += gridOffset ;
        out_ptr += outOffset ;
      }
      data_ptr += dataOffset ;
    }
    return VLE_Success ;
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
    static const std::string signature = std::string("BilinearSamplerBackwardCuDNN")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename DataTypeTraits<dataType>::type type ;
    Int inCardinality = derInput.getCardinality();
    Int inNumChannels = derInput.getNumChannels();
    Int inHeight = derInput.getHeight();
    Int inWidth = derInput.getWidth();
    Int outCardinality = derOutput.getCardinality();
    Int outNumChannels = derOutput.getNumChannels();
    Int outWidth = derOutput.getWidth();
    Int outHeight = derOutput.getHeight();
    Int groupSize = (int)(outCardinality / inCardinality) ;

    // Get CuDNN.
    cudnnHandle_t handle ;
    CKCUDNN(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors.
    CudnnTensorDescriptor derOutputDesc ;
    CKCUDNN(derOutputDesc.init(dataType,{outHeight,outWidth,outNumChannels,1})) ;

    CudnnTensorDescriptor dataDesc ;
    CKCUDNN(dataDesc.init(dataType,{inHeight,inWidth,inNumChannels,1})) ;

    // Get bilinear sampler descriptor.
    cudnnSpatialTransformerDescriptor_t samplerDesc ;
    CKCUDNN(cudnnCreateSpatialTransformerDescriptor(&samplerDesc)) ;
    auto transformerDescDeleter = defer([&]{cudnnDestroySpatialTransformerDescriptor(samplerDesc);}) ;
    int dimOut[4] = {1, (int)outNumChannels, (int)outWidth, (int)outHeight};
    CKCUDNN(cudnnSetSpatialTransformerNdDescriptor(samplerDesc,
                                                   CUDNN_SAMPLER_BILINEAR,
                                                   DataTypeToCudnn<dataType>::dataType,
                                                   4,
                                                   dimOut));

    // Perform calculations.
    type alpha = 1.0f ;
    type dataBeta = 1.0f ; // assuming that the derInputData has been initialized to zero
    type gridBeta = 0.0f ;
    Int dataOffset = inHeight * inWidth * inNumChannels ;
    Int gridOffset = 2 * outWidth * outHeight ;
    Int outOffset = outHeight * outWidth * outNumChannels ;
    type const* data_ptr = (type const*) input.getMemory() ;
    type * derInputData_ptr = (type *) derInput.getMemory() ;
    type const* grid_ptr = (type const*) grid.getMemory() ;
    type * derGrid_ptr = (type *) derGrid.getMemory() ;
    type * derOut_ptr = (type *) derOutput.getMemory() ;

    for (Int im=0; im < inCardinality; im++) {
      for (Int ig=0; ig < groupSize; ig++) {
        CKCUDNN(cudnnSpatialTfSamplerBackward(handle,
                                              samplerDesc,
                                              &alpha,
                                              dataDesc.get(), data_ptr,
                                              &dataBeta,
                                              dataDesc.get(), derInputData_ptr,
                                              &alpha,
                                              derOutputDesc.get(), derOut_ptr,
                                              grid_ptr,
                                              &gridBeta,
                                              derGrid_ptr)) ;
        grid_ptr += gridOffset ;
        derGrid_ptr += gridOffset ;
        derOut_ptr += outOffset ;
      }
      data_ptr += dataOffset ;
      derInputData_ptr += dataOffset ;
    }
    return VLE_Success ;
  }
} ;

#endif // CUDNN >= v5.0
