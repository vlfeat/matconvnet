// @file nnnormalizelp_gpu.cu
// @brief Batch normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnnormalizelp.hpp"
#include "datacu.hpp"
#include <vector>
#include <algorithm>

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

struct GPUVisitPattern
{
  size_t normsVolume ;
  size_t inputVolume ;
  int dims [4] {1,1,1,1} ;
  int strides [4] {0,0,0,0} ;
  int ndims [4] {1,1,1,1} ;
  int nstrides [4] {0,0,0,0} ;
} ;

GPUVisitPattern getGPUVisitPatternForInput(NormalizeLp const & op, vl::Tensor input)
{
  // Compute tensor geometry.
  int n = input.getNumDimensions() ;
  auto inputDimensions = std::vector<size_t>(input.getDimensions(),
                                             input.getDimensions() + n) ;

  assert(n <= 4) ; // Todo: relax.

  size_t inputVolume = 1 ;
  size_t normsVolume = 1 ;
  auto dims = std::vector<ptrdiff_t>{} ;
  auto steps = std::vector<ptrdiff_t>{} ;
  auto ndims = std::vector<ptrdiff_t>{} ;
  auto nstrides = std::vector<ptrdiff_t>{} ;

  // Find out how to traverse the reduced results as the input is
  // scanned from first to last element.
  for (int d = 0 ; d < n ; ++d) {
    bool squashed =
    (find(op.selectedDimensions.begin(), op.selectedDimensions.end(), d) !=
     op.selectedDimensions.end()) ;

    if (squashed) {
      dims.push_back(inputDimensions[d]) ;
      steps.push_back(inputVolume) ;
    } else {
      ndims.push_back(inputDimensions[d]) ;
      nstrides.push_back(inputVolume) ;
      normsVolume *= inputDimensions[d] ;
    }
    inputVolume *= inputDimensions[d] ;
  }

  //cout << steps.size() << " " << inputVolume << endl ;
  
  for (int d = steps.size() ; d < 5 ; ++d) {
    steps.push_back(inputVolume) ;
    dims.push_back(1) ;
  }
  for (int d = 3 ; d >= 0 ; d--) {
    steps[d+1] -= steps[d] * dims[d] ;
  }

  GPUVisitPattern vp ;
  vp.inputVolume = inputVolume ;
  vp.normsVolume = normsVolume ;
  std::copy(dims.begin(),dims.end(),vp.dims) ;
  std::copy(steps.begin(),steps.end(),vp.strides) ;
  std::copy(ndims.begin(),ndims.end(),vp.ndims) ;
  std::copy(nstrides.begin(),nstrides.end(),vp.nstrides) ;
  return vp ;
}

template<typename type> __global__ void
computeNorms(type * normsData,
             type const * inputData,
             type exponent,
             type epsilon,
             GPUVisitPattern vp)
{
  int tid = threadIdx.x ;
  if (tid >= vp.normsVolume) { return ; }
  normsData += tid ;

  int i0 = tid % vp.ndims[0] ; tid /= vp.ndims[0] ;
  int i1 = tid % vp.ndims[1] ; tid /= vp.ndims[1] ;
  int i2 = tid % vp.ndims[2] ; tid /= vp.ndims[2] ;
  int i3 = tid % vp.ndims[3] ;

  inputData +=
  i0 * vp.nstrides[0] +
  i1 * vp.nstrides[1] +
  i2 * vp.nstrides[2] +
  i3 * vp.nstrides[3] ;

  type value = 0 ;
  for (int i3 = 0 ; i3 < vp.dims[3] ; ++i3) {
    for (int i2 = 0 ; i2 < vp.dims[2] ; ++i2) {
      for (int i1 = 0 ; i1 < vp.dims[1] ; ++i1) {
        for (int i0 = 0 ; i0 < vp.dims[0] ; ++i0) {
          value = value + pow(*inputData, exponent) ;
          inputData += vp.strides[0] ;
        }
        inputData += vp.strides[1] ;
      }
      inputData += vp.strides[2] ;
    }
    inputData += vp.strides[3] ;
  }
  *normsData = pow(value + epsilon, static_cast<type>(1.0)/exponent) ;
}

template<typename type> __global__ void
divideByNorms(type * outputData,
              type const * inputData,
              type const * normsData,
              GPUVisitPattern vp)
{
  int tid = threadIdx.x ;
  if (tid >= vp.normsVolume) { return ; }
  normsData += tid ;

  int i0 = tid % vp.ndims[0] ; tid /= vp.ndims[0] ;
  int i1 = tid % vp.ndims[1] ; tid /= vp.ndims[1] ;
  int i2 = tid % vp.ndims[2] ; tid /= vp.ndims[2] ;
  int i3 = tid % vp.ndims[3] ;

  int offset =
  i0 * vp.nstrides[0] +
  i1 * vp.nstrides[1] +
  i2 * vp.nstrides[2] +
  i3 * vp.nstrides[3] ;

  inputData += offset ;
  outputData += offset ;

  type value = *normsData ;
  for (int i3 = 0 ; i3 < vp.dims[3] ; ++i3) {
    for (int i2 = 0 ; i2 < vp.dims[2] ; ++i2) {
      for (int i1 = 0 ; i1 < vp.dims[1] ; ++i1) {
        for (int i0 = 0 ; i0 < vp.dims[0] ; ++i0) {
          *outputData = *inputData / value ;
          inputData += vp.strides[0] ;
          outputData += vp.strides[0] ;
        }
        inputData += vp.strides[1] ;
        outputData += vp.strides[1] ;
      }
      inputData += vp.strides[2] ;
      outputData += vp.strides[2] ;
    }
    inputData += vp.strides[3] ;
    outputData += vp.strides[3] ;
  }
}

template<typename type> __global__ void
computeSum(type * scratchData,
           type const * inputData,
           type const * derOutputData,
           GPUVisitPattern vp)
{
  int tid = threadIdx.x ;
  if (tid >= vp.normsVolume) { return ; }
  scratchData += tid ;

  int i0 = tid % vp.ndims[0] ; tid /= vp.ndims[0] ;
  int i1 = tid % vp.ndims[1] ; tid /= vp.ndims[1] ;
  int i2 = tid % vp.ndims[2] ; tid /= vp.ndims[2] ;
  int i3 = tid % vp.ndims[3] ;

  int offset =
  i0 * vp.nstrides[0] +
  i1 * vp.nstrides[1] +
  i2 * vp.nstrides[2] +
  i3 * vp.nstrides[3] ;

  inputData += offset ;
  derOutputData += offset ;

  type value = 0 ;
  for (int i3 = 0 ; i3 < vp.dims[3] ; ++i3) {
    for (int i2 = 0 ; i2 < vp.dims[2] ; ++i2) {
      for (int i1 = 0 ; i1 < vp.dims[1] ; ++i1) {
        for (int i0 = 0 ; i0 < vp.dims[0] ; ++i0) {
          value += (*inputData) * (*derOutputData) ;
          inputData += vp.strides[0] ;
          derOutputData += vp.strides[0] ;
        }
        inputData += vp.strides[1] ;
        derOutputData += vp.strides[1] ;
      }
      inputData += vp.strides[2] ;
      derOutputData += vp.strides[2] ;
    }
    inputData += vp.strides[3] ;
    derOutputData += vp.strides[3] ;
  }
  *scratchData = value ;
}


template<typename type> __global__ void
computeDerInput(type * derInputData,
                type const * inputData,
                type const * normsData,
                type const * derOutputData,
                type const * scratchData,
                type exponent,
                GPUVisitPattern vp)
{
  int tid = threadIdx.x ;
  if (tid >= vp.normsVolume) { return ; }
  normsData += tid ;
  scratchData += tid ;

  int i0 = tid % vp.ndims[0] ; tid /= vp.ndims[0] ;
  int i1 = tid % vp.ndims[1] ; tid /= vp.ndims[1] ;
  int i2 = tid % vp.ndims[2] ; tid /= vp.ndims[2] ;
  int i3 = tid % vp.ndims[3] ;

  int offset =
  i0 * vp.nstrides[0] +
  i1 * vp.nstrides[1] +
  i2 * vp.nstrides[2] +
  i3 * vp.nstrides[3] ;

  derInputData += offset ;
  inputData += offset ;
  derOutputData += offset ;

  type const nv = *normsData ;
  type const sv = *scratchData ;

  for (int i3 = 0 ; i3 < vp.dims[3] ; ++i3) {
    for (int i2 = 0 ; i2 < vp.dims[2] ; ++i2) {
      for (int i1 = 0 ; i1 < vp.dims[1] ; ++i1) {
        for (int i0 = 0 ; i0 < vp.dims[0] ; ++i0) {
          type iv = *inputData ;
          type dov = *derOutputData ;

          *derInputData = dov / nv - sv * pow(iv,exponent-1) / pow(nv,exponent+1) ;

          derInputData += vp.strides[0] ;
          inputData += vp.strides[0] ;
          derOutputData += vp.strides[0] ;
        }
        derInputData += vp.strides[1] ;
        inputData += vp.strides[1] ;
        derOutputData += vp.strides[1] ;
      }
      derInputData += vp.strides[2] ;
      inputData += vp.strides[2] ;
      derOutputData += vp.strides[2] ;
    }
    derInputData += vp.strides[3] ;
    inputData += vp.strides[3] ;
    derOutputData += vp.strides[3] ;
  }
}

// -------------------------------------------------------------------
//                                                         GPU forward
// -------------------------------------------------------------------

template<vl::DataType dataType, bool givenNorms>
struct NormalizeLpForwardGPU
{
  vl::ErrorCode operator()(NormalizeLp & op,
                           Tensor &output,
                           typename NormAgrument<givenNorms>::type norms,
                           Tensor const &input)
  {
    assert(norms || !givenNorms) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto vp = getGPUVisitPatternForInput(op,input) ;

    // Get buffers.
    type const * inputData = (type const*)input.getMemory() ;
    type * normsData ;
    if (norms) {
      normsData = (type*)norms.getMemory() ;
    }
    else {
      normsData = (type*)op.context.getWorkspace
      (vl::VLDT_GPU, vp.normsVolume * sizeof(type)) ;
    }

    // Accumulate norms.
    if (!givenNorms) {
      computeNorms<type>
      <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (normsData,inputData,op.exponent,op.epsilon,vp) ;
    }

    // Divide by them.
    type * outputData = (type*)output.getMemory() ;
    divideByNorms<type>
    <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (outputData,inputData,normsData,vp) ;

    //cout << "n vol " << vp.normsVolume << endl ;
    return vl::VLE_Success ;
  }
} ;

template<vl::DataType dataType>
struct NormalizeLpForward<vl::VLDT_GPU, dataType>
: public NormalizeLpForwardGPU<dataType,false>
{ } ;

template<vl::DataType dataType>
struct NormalizeLpForwardWithNorms<vl::VLDT_GPU, dataType>
: public NormalizeLpForwardGPU<dataType,true>
{ } ;

// -------------------------------------------------------------------
//                                                        GPU backward
// -------------------------------------------------------------------

template<vl::DataType dataType, bool givenNorms>
struct NormalizeLpBackwardGPU
{
  vl::ErrorCode operator()(NormalizeLp &op,
                           Tensor &derInput,
                           typename NormAgrument<givenNorms>::type norms,
                           Tensor const &input,
                           Tensor const& derOutput)
  {
    assert(norms || !givenNorms) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto vp = getGPUVisitPatternForInput(op,input) ;

    // Get buffers.
    size_t workspaceSize = vp.normsVolume * sizeof(type) ;
    type const * inputData = (type const*)input.getMemory() ;
    type * normsData ;
    if (norms) {
      normsData = (type*)norms.getMemory() ;
    }
    else {
      normsData = 0 ;
      workspaceSize *= 2 ;
    }
    type * scratchData = (type*)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize) ;
    if (normsData == NULL) {
      normsData = scratchData + vp.normsVolume ;
    }

    // Accumulate norms.
    if (!givenNorms) {
      computeNorms<type>
      <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (normsData,inputData,op.exponent,op.epsilon,vp) ;
    }

    // Compute sum(derOutput .* input).
    type const* derOutputData = (type const*)derOutput.getMemory() ;
    computeSum<type>
    <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (scratchData,inputData,derOutputData,vp) ;

    // Compute derInputs.
    type * derInputData = (type*)derInput.getMemory() ;
    computeDerInput<type>
    <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (derInputData,inputData,normsData,derOutputData,scratchData,op.exponent,vp) ;

    return vl::VLE_Success ;
  }
} ;

template<vl::DataType dataType>
struct NormalizeLpBackward<vl::VLDT_GPU, dataType>
: public NormalizeLpBackwardGPU<dataType,false>
{ } ;

template<vl::DataType dataType>
struct NormalizeLpBackwardWithNorms<vl::VLDT_GPU, dataType>
: public NormalizeLpBackwardGPU<dataType,true>
{ } ;
