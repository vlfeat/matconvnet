// @file bilinearsampler_gpu.cu
// @brief Bilinear sampler CUDA implementation
// @author Ankush Gupta
// @author Andrea Vedaldi

/*
Copyright (C) 2016-17 Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../datacu.hpp"
#include <cassert>

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

// maximum size of each grid dimension:
#define MAX_GRID_DIM 65535 // this is probably a bad idea..

/* 2D grid of 1D blocks. */
__device__ int getGlobalIdx_2D_1D()
{
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId ;
}

// todo: fix such assumptions either in doc or by clearing memory
// probably all these functions should have the option to accumulate, so...
// assumption: derInputData is cleared before calling this code

template<typename type, bool backwardData>
__global__ void forward_backward_kernel
(type* output,
 type* derInputData,
 type const* data,
 type const* grid,
 type const* derOutput,
 int outHeight, int outWidth, int outNumChannels, int outCardinality,
 int inHeight, int inWidth, int inCardinality)
{
  const int offset = getGlobalIdx_2D_1D();
  const int nOut = outWidth * outHeight * outNumChannels * outCardinality ;
  if (offset >= nOut) { return ; }
  bool backward = backwardData;

  // get the index of the output image, feature channel, and pixel
  int k = offset ;
  int c = k / (outHeight * outWidth) ; 
  int n = c / outNumChannels ; // out image index
  k %= (outHeight * outWidth) ; // out spatial index
  c %= outNumChannels ; // out channel index

  // get the index of the input image
  int groupSize = outCardinality / inCardinality ; // num of transformations/image
  int nInputImage = n / groupSize ; // index of the input image
  int inputOffset = (inHeight * inWidth)*(outNumChannels * nInputImage + c) ; // location of the start of the input image
  int gridOffset = 2 * ((outHeight * outWidth) * n + k) ; //+ 1;    // location of the first grid coordinate for this output pixel
  //int gridOffset = 2*k+1 ;

  // get the grid for this output image
  type py = grid[gridOffset + 0] ;
  type px = grid[gridOffset + 1] ;

  py = type(0.5)*(py + type(1.0)) * (inHeight - 1) ;
  px = type(0.5)*(px + type(1.0)) * (inWidth - 1) ;
  const int sx = floor(px); // todo: check floor vs floorf
  const int sy = floor(py);

  type acc = 0 ;
  type dy ;
  if (!backward) {
    data += inputOffset ;
  }
  if (backwardData) {
    derInputData += inputOffset ;
  }
  if (backward) {
    dy = derOutput[offset] ;
  }

  if (-1 <= sy && sy < inHeight && -1 <= sx && sx < inWidth) {
    // get the interpolation weights
    const type wx = px - sx ;
    const type wy = py - sy ;

    #pragma unroll
    for (int j=0; j < 2; j++) {
      #pragma unroll
      for (int i=0; i < 2; i++) {
        int ssy = sy + i ;
        int ssx = sx + j ;
        if (ssy < 0 || ssy >= inHeight || ssx < 0 || ssx >= inWidth) {
          continue ;
        }
        type wwx = (1-j)*(1-wx) + j*wx ;
        type wwy = (1-i)*(1-wy) + i*wy ;
        type ww = wwx * wwy ;
        if (!backward) {
          acc += ww * data[ssy + ssx * inHeight];
        } else {
          if (backwardData) {
            atomicAdd(derInputData  + ssy + ssx * inHeight, ww * dy) ;
          }
        }
      }
    }
    if (!backward) {
      output[offset] = acc ;
    }
  }
}

template<typename type>
__global__ void grid_backward_kernel
(type* derGrid,
 type const* data,
 type const* grid,
 type const* derOutput,
 int outHeight, int outWidth, int outNumChannels, int outCardinality,
 int inHeight, int inWidth, int inCardinality)
{
  const int offset = getGlobalIdx_2D_1D();
  const int nOut = outWidth * outHeight * outCardinality ;
  if (offset >= nOut) { return ; }

  // get the index of the output image, feature channel, and pixel
  int k = offset ;
  int n = k / (outHeight * outWidth) ; // out image index
  k %= (outHeight * outWidth) ; // out spatial index

  // get the grid offset:
  //  --> location of the first grid coordinate for this output pixel
  int gridOffset = 2 * ((outHeight * outWidth) * n + k) ; //+ 1;  

  // get the index of the input image
  const int groupSize = outCardinality / inCardinality ; // num of transformations/image
  const int nInputImage = n / groupSize ; // index of the input image
  const int inputOffset = inHeight * inWidth * outNumChannels * nInputImage ; // location of the start of the input image

  // get the grid for this output image
  type py = grid[gridOffset + 0] ;
  type px = grid[gridOffset + 1] ;

  py = type(0.5)*(py + type(1.0)) * (inHeight - 1) ;
  px = type(0.5)*(px + type(1.0)) * (inWidth - 1) ;
  const int sx = floor(px); // todo: check floor vs floorf
  const int sy = floor(py);

  type dgridx = 0 ;
  type dgridy = 0 ;
  data += inputOffset ;
  derOutput += k + n * outWidth * outHeight * outNumChannels ;

  if (-1 <= sy && sy < inHeight && -1 <= sx && sx < inWidth) {
    // get the interpolation weights
    const type wx = px - sx ;
    const type wy = py - sy ;

    #pragma unroll
    for (int j=0; j < 2; j++) {
      #pragma unroll
      for (int i=0; i < 2; i++) {
        int ssy = sy + i ;
        int ssx = sx + j ;
        if (ssy < 0 || ssy >= inHeight || ssx < 0 || ssx >= inWidth) {
          continue ;
        }
        const type wwx = (2*i-1) * ( (1-j)*(1-wx) + j*wx ) ;
        const type wwy = (2*j-1) * ( (1-i)*(1-wy) + i*wy ) ;
        for (int ic=0; ic < outNumChannels; ic++) {
          const type dy = derOutput[ic * outHeight * outWidth];
          const type x = data[ssy  +  ssx * inHeight  +  ic * inHeight * inWidth];
          dgridy += wwx * dy * x ;
          dgridx += wwy * dy * x ;
        }
      }
    }
    derGrid[gridOffset + 0] = type(0.5)*(inHeight - 1) * dgridy ;
    derGrid[gridOffset + 1] = type(0.5)*(inWidth - 1) * dgridx ;
  }
}

/** get the number of threads (1D) and blocks (2D). **/
vl::ErrorCode get_launch_params(Int const& N, Int& nTh, Int& nGx, Int& nGy)
{
  nGx = (int)vl::divideAndRoundUp((unsigned)N,VL_CUDA_NUM_THREADS);
  if (nGx == 1) {
    nTh = N;
    nGy = 1;
  } else {
    nTh = VL_CUDA_NUM_THREADS;
    if (nGx <= MAX_GRID_DIM) {
      nGy = 1;
    } else {
      nGy = vl::divideAndRoundUp(nGx, (Int)MAX_GRID_DIM);
      nGx = MAX_GRID_DIM;
      if (nGy > MAX_GRID_DIM) {
        // the following print statement is probably not
        // shown in the matlab JVM console:
        std::printf("BilinearSamper: output volume should be smaller.");
        return vl::VLE_Cuda;
      }
    }
  }
  return vl::VLE_Success;
}

// use a template to define both directions as they are nearly identical code-wise
template<typename type, bool backwardData, bool backwardGrid>
static vl::ErrorCode
forward_backward_gpu
(vl::Context& context,
 type* output,
 type* derInputData,
 type* derGrid,
 type const* data,
 type const* grid,
 type const* derOutput,
 Int outHeight, Int outWidth, Int outNumChannels, Int outCardinality,
 Int inHeight, Int inWidth, Int inCardinality)
{
  //bool backward = backwardData || backwardGrid ;
  // common conditions
  assert(grid) ;
  assert(divides(inCardinality, outCardinality)) ;

  // forward conditions
  //assert(backward || data) ;
  //assert(backward || output) ;

  // backward conditions
  //assert(!backward || derOutput) ;
  assert(!backwardData || derInputData) ;
  assert(!backwardGrid || derGrid) ;
  assert(!backwardGrid || data) ;

  // if (backwardData) {
  //   //memset(derInputData, 0, inHeight * inWidth * outNumChannels * inCardinality * sizeof(type)) ;
  // }

  // setup and launch the kernel for DER-DATA:
  Int nTh, nGx, nGy;
  Int outVolume = outHeight * outWidth * outNumChannels * outCardinality ;
  vl::ErrorCode volume_ok = get_launch_params(outVolume, nTh, nGx, nGy);
  if (volume_ok != vl::VLE_Success) { return volume_ok; }

  dim3 gridDim((unsigned)nGx,(unsigned)nGy); // grid-dimensions
  forward_backward_kernel <type, backwardData>
    <<< gridDim, (unsigned)nTh >>>
  (output,
   derInputData,
   data,
   grid,
   derOutput,
   (int)outHeight, (int)outWidth, (int)outNumChannels, (int)outCardinality,
   (int)inHeight, (int)inWidth, (int)inCardinality) ;

  auto error = context.getCudaHelper().catchCudaError(__func__) ;
  if (error != vl::VLE_Success) {
    return context.setError(error) ;
  }

  if (backwardGrid) {
    // setup and launch kernel for DER-GRID:
    auto const outN = outHeight * outWidth * outCardinality;
    volume_ok = get_launch_params(outN, nTh, nGx, nGy);
    if (volume_ok != vl::VLE_Success) { return volume_ok;}

    gridDim.x = (unsigned)nGx;
    gridDim.y = (unsigned)nGy;
    grid_backward_kernel <type>
    <<< gridDim, (unsigned)nTh >>>
    (derGrid,
     data, grid,
     derOutput,
     (int)outHeight, (int)outWidth, (int)outNumChannels, (int)outCardinality,
     (int)inHeight, (int)inWidth, (int)inCardinality ) ;

    auto error = context.getCudaHelper().catchCudaError(__func__) ;
    if (error != VLE_Success) {
      return context.setError(error) ;
    }
  }
  return vl::VLE_Success ;
}

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BilinearSamplerForward<VLDT_GPU,dataType>
{
  vl::ErrorCode operator()
  (BilinearSampler const &op,
   Tensor &output,
   Tensor const &input,
   Tensor const &grid)
  {
    static const std::string signature = std::string("BilinearSamplerForward[MCN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename DataTypeTraits<dataType>::type type ;
    return op.getContext().passError
    (forward_backward_gpu<type, false, false>
     (op.getContext(),
      (type*)output.getMemory(), NULL, NULL,
      (type const*)input.getMemory(),
      (type const*)grid.getMemory(), NULL,
      (int)output.getHeight(), (int)output.getWidth(),
      (int)output.getNumChannels(), (int)output.getCardinality(),
      (int)input.getHeight(), (int)input.getWidth(), (int)input.getCardinality()),
     signature.c_str()) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

#undef DISPATCH
#define DISPATCH(bwData, bwGrid) \
error = forward_backward_gpu<type, bwData, bwGrid> \
    (op.getContext(), NULL, derInputData, derGridData, inputData, gridData, derOutputData, \
     outHeight, outWidth, outNumChannels, outCardinality, \
     inHeight, inWidth,inCardinality) ;

template<DataType dataType>
struct BilinearSamplerBackward<VLDT_GPU,dataType>
{
  vl::ErrorCode operator()
  (BilinearSampler const &op,
   Tensor &derInput,
   Tensor &derGrid,
   Tensor const &input,
   Tensor const &grid,
   Tensor const &derOutput)
  {
    vl::ErrorCode error = VLE_Success ;

    static const std::string signature = std::string("BilinearSamplerForward[MCN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename DataTypeTraits<dataType>::type type ;
    Int outHeight = derOutput.getHeight() ;
    Int outWidth = derOutput.getWidth() ;
    Int outNumChannels = derOutput.getNumChannels() ;
    Int outCardinality = derOutput.getCardinality() ;
    Int inHeight = input.getHeight() ;
    Int inWidth = input.getWidth() ;
    Int inCardinality = input.getCardinality() ;

    auto derInputData = (type*)derInput.getMemory() ;
    auto derGridData = (type*)derGrid.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto gridData = (type const*)grid.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

    // optimized codepaths depending on what needs to be comptued
    if (derInput && !derGrid) {
      DISPATCH(true, false) ;
    } else if (!derInput && derGrid) {
      DISPATCH(false, true) ;
    } else if (derInput && derGrid) {
      DISPATCH(true, true) ;
    }
    return op.getContext().passError(error,signature.c_str()) ;
  }
} ;


