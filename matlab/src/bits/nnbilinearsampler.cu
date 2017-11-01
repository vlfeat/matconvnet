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
#include <sstream>

using namespace std ;
using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct BilinearSamplerForward ;
template<DeviceType deviceType, DataType dataType> struct BilinearSamplerBackward ;
template<DataType dataType> struct BilinearSamplerForwardCudnn ;
template<DataType dataType> struct BilinearSamplerBackwardCudnn ;

#if ENABLE_GPU
#include "impl/nnbilinearsampler_gpu.cu"
#endif

#if ENABLE_CUDNN
#include "impl/nnbilinearsampler_cudnn.cu"
#endif

// -------------------------------------------------------------------
/// MARK: - Helpers
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
 Int outHeight, Int outWidth, Int outNumChannels, Int outCardinality,
 Int inHeight, Int inWidth, Int inCardinality)
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

  auto groupSize = outCardinality / inCardinality ;

  // don't need these -- as already being initialized with zeros in the mex file:
  // if (backwardData) {
  //   memset(derData, 0, inHeight * inWidth * outNumChannels * inCardinality * sizeof(type)) ;
  // }
  // if (backwardGrid) {
  //   memset(derGrid, 0, 2 * outHeight * outWidth * outCardinality * sizeof(type)) ;
  // }
  for (Int n = 0 ; n < (signed)outCardinality ; ++n) {
    for (Int c = 0 ; c < (signed)outNumChannels ; ++c) {
      type const * end = grid + 2 * outWidth * outHeight ;
      while (grid < end) {
        type py = *grid++ ;
        type px = *grid++ ;

        py = type(0.5)*(py + type(1.0)) * (inHeight - 1) ;
        px = type(0.5)*(px + type(1.0)) * (inWidth - 1) ;
        auto const sx = (Int)floor(px); // todo: check floor vs floorf
        auto const sy = (Int)floor(py);

        type acc = 0 ;
        type dgridx = 0 ;
        type dgridy = 0 ;
        type dy = 0 ;
        if (backward) {
          dy = *derOutput++ ;
        }

        // todo: check boundary conditions in other frameworks and make
        // them the same
        if (-1 <= sy && sy < (signed)inHeight && -1 <= sx && sx < (signed)inWidth) {
          // get the interpolation weights
          const type wx = px - sx ;
          const type wy = py - sy ;

#pragma unroll
          for (Int j=0; j < 2; j++) {
#pragma unroll
            for (Int i=0; i < 2; i++) {
              auto ssy = sy + i ;
              auto ssx = sx + j ;
              if (ssy < 0 || ssy >= (signed)inHeight || ssx < 0 || ssx >= (signed)inWidth) {
                continue ;
              }
              type wwx = (1-j)*(1-wx) + j*wx ;
              type wwy = (1-i)*(1-wy) + i*wy ;
              type ww = wwx * wwy ;
              if (!backward) {
                acc += ww * data[ssy + ssx * (signed)inHeight];
              } else {
                if (backwardData) {
                  derData[ssy + ssx * (signed)inHeight] += ww * dy ;
                }
                if (backwardGrid) {
                  type x = data[ssy + ssx * (signed)inHeight] ;
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
    if ((as_unsigned(n) + 1) % as_unsigned(groupSize) != 0) {
      data -= inHeight * inWidth * outNumChannels ;
      derData -= inHeight * inWidth * outNumChannels ;
    }
    grid += 2 * outHeight * outWidth ;
    derGrid += 2 * outHeight * outWidth ;
  }
  return error ;
}

// -------------------------------------------------------------------
/// MARK: - Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BilinearSamplerForward<VLDT_CPU,dataType>
{
  vl::ErrorCode operator()
  (BilinearSampler const &op,
   Tensor &output,
   Tensor const &input,
   Tensor const &grid)
  {
    static const std::string signature = std::string("BilinearSamplerForward[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    typedef typename DataTypeTraits<dataType>::type type ;

    Int outHeight = output.getHeight() ;
    Int outWidth = output.getWidth() ;
    Int outNumChannels = output.getNumChannels() ;
    Int outCardinality = output.getCardinality() ;
    Int inHeight = input.getHeight() ;
    Int inWidth = input.getWidth() ;
    Int inCardinality = input.getCardinality() ;

    auto outputData = (type*)output.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto gridData = (type const*)grid.getMemory() ;

    return op.getContext().passError
    (forward_backward<type, false, false>
     (op.getContext(), outputData, NULL, NULL, inputData, gridData, NULL,
      outHeight, outWidth, outNumChannels, outCardinality,
      inHeight, inWidth, inCardinality),
     signature.c_str());
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Backward
// -------------------------------------------------------------------

#undef DISPATCH
#define DISPATCH(bwData, bwGrid) \
error = forward_backward<type, bwData, bwGrid> \
(op.getContext(), NULL, derInputData, derGridData, inputData, gridData, derOutputData, \
outHeight, outWidth, outNumChannels, outCardinality, \
inHeight, inWidth,inCardinality) ;

template<DataType dataType>
struct BilinearSamplerBackward<VLDT_CPU,dataType>
{
  vl::ErrorCode operator()
  (BilinearSampler const &op,
   Tensor &derInput,
   Tensor &derGrid,
   Tensor const &input,
   Tensor const &grid,
   Tensor const &derOutput)
  {
    static const std::string signature = std::string("BilinearSamplerBacwkard[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature.c_str() ;

    typedef typename DataTypeTraits<dataType>::type type ;
    auto outHeight = derOutput.getHeight() ;
    auto outWidth = derOutput.getWidth() ;
    auto outNumChannels = derOutput.getNumChannels() ;
    auto outCardinality = derOutput.getCardinality() ;
    auto inHeight = input.getHeight() ;
    auto inWidth = input.getWidth() ;
    auto inCardinality = input.getCardinality() ;
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
    return op.getContext().passError(error,signature.c_str()) ;
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Driver
// -------------------------------------------------------------------

BilinearSampler::BilinearSampler(Context &context)
: Operation(context)
{ }

vl::ErrorCode
BilinearSampler::forwardShape(TensorShape &output,
                              TensorShape const &input,
                              TensorShape const &grid) const
{
  Int inNumChannels = input.getNumChannels();
  Int inBatch = input.getCardinality();

  // Grid uses the first dimension has channels.
  Int gridDepth = grid.getDimension(0);
  Int gridHeight = grid.getDimension(1);
  Int gridWidth = grid.getDimension(2);
  Int gridBatch = grid.getDimension(3);

  output.clear() ; // make empty

  if (gridDepth != 2) {
    auto message = std::ostringstream()<<
    "BilinearSamplerForwardShape: GRID has " << gridDepth << " channels instead of 2." ;
    return getContext().setError
    (VLE_TensorShapeMismatch, message.str().c_str()) ;
  }

  if ((gridBatch % inBatch) != 0) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "BilinearSamplerForwardShape: The cardinality of GRID is not a multiple of the cardinality of DATA.") ;
  }

  output = TensorShape(gridHeight, gridWidth, inNumChannels, gridBatch) ;
  return VLE_Success ;
}

vl::ErrorCode
BilinearSampler::forward(Tensor &output,
                         Tensor const &input,
                         Tensor const &grid) const
{
  ErrorCode error ;

  // Validate arguments.
  if (!check_tensor_compatibility(output,input,grid)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "BilinearSamplerForward: the tensors have mismatching data or device type.") ;
  }
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape,input,grid)) != VLE_Success) {
    return error ;
  }
  if (output != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "BilinearSamplerForward: OUTPUT does not have the appropriate dimensions.") ;
  }
  if (input.isEmpty() | input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "BilinearSamplerForward: INPUT is empty or has no data.") ;
  }
  if (output.isEmpty() | output.isNull()) {
    return  getContext().setError
    (VLE_IllegalArgument,
     "BilinearSamplerForward: OUTPUT is empty or has no data.") ;
  }

  VLLOG(*this,1)
  << "BilinearSamplerForward: input=" << pretty(input.getDimensions())
  << " gird=" << pretty(grid.getDimensions()) ;

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
                          Tensor const &derOutput) const
{
  // Validate arguments.
  ErrorCode error ;

  if (!check_tensor_compatibility(derInput,derGrid,input,grid,derOutput)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "BilinearSamplerBackward: the tensors have mismatching data or device type.") ;
  }

  // Check that we have the output derivative.
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape,input,grid)) != VLE_Success) {
    return error ;
  }
  if (derOutput != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "BilinearSamplerBackward: DEROUTPUT does not have the appropriate dimensions.") ;
  }
  if (derOutput.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "BilinearSamplerBackward: DEROUTPUT is null.") ;
  }

  // The grid is needed for both DERINPUT and DERGRID.
  if (grid.isEmpty() || grid.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "BilinearSamplerBackward: GRID is either empty or null.") ;
  }

  // If the input derivatives are requested, check that we have what we need.
  if (!derInput.isEmpty()) {
    if (derInput.isNull()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "BilinearSamplerBackward: DERINPUT requested, but the tensor is null.") ;
    }
    if (static_cast<TensorShape>(derInput) != static_cast<TensorShape>(input)) {
      return getContext().setError
      (VLE_IllegalArgument,
       "BilinearSamplerBackward: DERINPUT requested, but its size is not the same as INPUT.") ;
    }
  }

  // If the grid derivatives are requested, check that we have what we need.
  if (!derGrid.isEmpty()) {
    if (derGrid.isNull()) {
      return getContext().setError
      (VLE_IllegalArgument,
       "BilinearSamplerBackward: DERGRID requested, but the tensor is null.") ;
    }
    if (static_cast<TensorShape>(derGrid) != static_cast<TensorShape>(grid)) {
      return getContext().setError
      (VLE_IllegalArgument,
       "BilinearSamplerBackward: DERGRID requested, but its size is not the same as GRID.") ;
    }
  }

  // Arguments are sane here.
  return dispatch_cudnn<
  BilinearSamplerBackward,
  BilinearSamplerBackwardCudnn>()
  (*this,derInput,derGrid,input,grid,derOutput) ;
}
