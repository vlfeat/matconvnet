// @file nnbilinearsampler.cu
// @brief Bilinear sampler block
// @author Ankush Gupta
// @author Andrea Vedaldi

/*
Copyright (C) 2016-17 Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbilinearsampler.hpp"
#include "impl/dispatcher.hpp"
#include <cassert>
#include <cmath>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct BilinearSamplerForward ;
template<DeviceType deviceType, DataType dataType> struct BilinearSamplerBackward ;
template<DataType dataType> struct BilinearSamplerForwardCudnn ;
template<DataType dataType> struct BilinearSamplerBackwardCudnn ;

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

template<typename type, bool backwardData, bool backwardGrid>
static vl::ErrorCode
forward_backward
(vl::Context& context,
 type* output,
 type* derData,
 type* derGrid,
 type const* data,
 type const* grid,
 type const* derOutput,
 size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
 size_t inHeight, size_t inWidth, size_t inCardinality)
{
  vl::ErrorCode error = vl::VLE_Success ;

  bool backward = backwardData | backwardGrid ;

  // common conditions
  assert(grid) ;
  assert(divides(inCardinality, outCardinality)) ;

  // forward conditions
  assert(backward || data) ;
  assert(backward || output) ;

  // backward conditions
  assert(!backward || derOutput) ;
  assert(!backwardData || derData) ;
  assert(!backwardGrid || derGrid) ;
  assert(!backwardGrid || data) ;

  int groupSize = outCardinality / inCardinality ;

  // don't need these -- as already being initialized with zeros in the mex file:
  // if (backwardData) {
  //   memset(derData, 0, inHeight * inWidth * outDepth * inCardinality * sizeof(type)) ;
  // }
  // if (backwardGrid) {
  //   memset(derGrid, 0, 2 * outHeight * outWidth * outCardinality * sizeof(type)) ;
  // }
  for (int n = 0 ; n < outCardinality ; ++n) {
    for (int c = 0 ; c < outDepth ; ++c) {
      type const * end = grid + 2 * outWidth * outHeight ;
      while (grid < end) {
        type py = *grid++ ;
        type px = *grid++ ;

        py = type(0.5)*(py + type(1.0)) * (inHeight - 1) ;
        px = type(0.5)*(px + type(1.0)) * (inWidth - 1) ;
        const int sx = floor(px); // todo: check floor vs floorf
        const int sy = floor(py);

        type acc = 0 ;
        type dgridx = 0 ;
        type dgridy = 0 ;
        type dy ;
        if (backward) {
          dy = *derOutput++ ;
        }

        // todo: check boundary conditions in other frameworks and make
        // them the same
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
                  derData[ssy + ssx * inHeight] += ww * dy ;
                }
                if (backwardGrid) {
                  type x = data[ssy + ssx * inHeight] ;
                  dgridx += (2*j-1) * wwy * dy * x ;
                  dgridy += (2*i-1) * wwx * dy * x ;
                }
              }
            }
          }
        }
        if (!backward) {
          *output++ = acc ;
        }
        if (backwardGrid) {
          *derGrid++ += type(0.5)*(inHeight - 1) * dgridy ;
          *derGrid++ += type(0.5)*(inWidth - 1) * dgridx ;
        }
      }
      // next channel
      data += inHeight * inWidth ;
      derData +=inHeight * inWidth ;
      grid -= 2 * outHeight * outWidth ;
      derGrid -= 2 * outHeight * outWidth ;
    }
    // next image
    if ((n + 1) % groupSize != 0) {
      data -= inHeight * inWidth * outDepth ;
      derData -= inHeight * inWidth * outDepth ;
    }
    grid += 2 * outHeight * outWidth ;
    derGrid += 2 * outHeight * outWidth ;
  }
  return error ;
}

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BilinearSamplerForward<VLDT_CPU,dataType>
{
  vl::ErrorCode operator()
  (BilinearSampler &op,
   Tensor &output,
   Tensor const &input,
   Tensor const &grid)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    auto outHeight = output.getHeight() ;
    auto outWidth = output.getWidth() ;
    auto outDepth = output.getDepth() ;
    auto outCardinality = output.getSize() ;
    auto inHeight = input.getHeight() ;
    auto inWidth = input.getWidth() ;
    auto inCardinality = input.getSize() ;
    auto outputData = (type*)output.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto gridData = (type const*)grid.getMemory() ;

    return forward_backward<type, false, false>
    (op.context, outputData, NULL, NULL, inputData, gridData, NULL,
     outHeight, outWidth, outDepth, outCardinality,
     inHeight, inWidth, inCardinality) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

#define DISPATCH(bwData, bwGrid) \
error = forward_backward<type, bwData, bwGrid> \
(op.context, NULL, derInputData, derGridData, inputData, gridData, derOutputData, \
outHeight, outWidth, outDepth, outCardinality, \
inHeight, inWidth,inCardinality) ;

template<DataType dataType>
struct BilinearSamplerBackward<VLDT_CPU,dataType>
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
    auto outHeight = derOutput.getHeight() ;
    auto outWidth = derOutput.getWidth() ;
    auto outDepth = derOutput.getDepth() ;
    auto outCardinality = derOutput.getSize() ;
    auto inHeight = input.getHeight() ;
    auto inWidth = input.getWidth() ;
    auto inCardinality = input.getSize() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derGridData = (type*)derGrid.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto gridData = (type const*)grid.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

    vl::ErrorCode error = VLE_Success ;

    // optimized codepaths depending on what needs to be comptued
    if (derInput && !derGrid) {
      DISPATCH(true, false) ;
    } else if (!derInputData && derGridData) {
      DISPATCH(false, true) ;
    } else if (derInputData && derGridData) {
      DISPATCH(true, true) ;
    }
    return error ;
  }
} ;

// -------------------------------------------------------------------
//                                                             Drivers
// -------------------------------------------------------------------

#if ENABLE_GPU
#include "nnbilinearsampler_gpu.cu"
#endif

#if ENABLE_CUDNN
#include "nnbilinearsampler_cudnn.cu"
#endif

BilinearSampler::BilinearSampler(Context &context)
: context(context)
{ }

vl::ErrorCode
BilinearSampler::forward(Tensor &output,
                         Tensor const &input,
                         Tensor const &grid)
{
  return dispatch_cudnn<
  BilinearSamplerForward,
  BilinearSamplerForwardCudnn>()
  (*this,output,input,grid) ;
}

vl::ErrorCode
BilinearSampler::backward(Tensor &derInput,
                          Tensor &derGrid,
                          Tensor const &input,
                          Tensor const &grid,
                          Tensor const &derOutput)
{
  return dispatch_cudnn<
  BilinearSamplerBackward,
  BilinearSamplerBackwardCudnn>()
  (*this,derInput,derGrid,input,grid,derOutput) ;
}
