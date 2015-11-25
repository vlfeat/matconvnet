// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnconv.hpp"
#include "nnbias.hpp"
#include "impl/nnconv_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnconv_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

/*
 for output: must have data and optional filters or biases
 */

vl::Error
vl::nnconv_forward(Context& context,
                   Tensor output, double outputMult,
                   Tensor data, double dataMult,
                   Tensor filters,
                   Tensor biases,
                   int strideY, int strideX,
                   int padTop, int padBottom,
                   int padLeft, int padRight)
{
  vl::Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::nnconv_forward_blas<CPU,float>
      (context,
       output, outputMult,
       data, dataMult,
       filters, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        status = vl::impl::nnconv_forward_cudnn<float>
        (context,
         output, outputMult,
         data, dataMult,
         filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnconv_forward_blas<GPU,float>
      (context,
       output, outputMult,
       data, dataMult,
       filters, biases,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return status ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */


/*
 for derBiases:  must have derOuptut
 for derData:    must have derData, derOutput and filters
 for derFilters: must have derFilters, derOutput and data
 */

vl::Error
vl::nnconv_backward(Context& context,
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
  vl::Error status = vl::vlSuccess ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      status = vl::impl::nnconv_backward_blas<CPU,float>
      (context,
       derData, derFilters, derBiases,
       data, filters, derOutput,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        status = vl::impl::nnconv_backward_cudnn<float>
        (context,
         derData, derFilters, derBiases,
         data, filters, derOutput,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      status = vl::impl::nnconv_backward_blas<GPU,float>
      (context,
       derData, derFilters, derBiases,
       data, filters, derOutput,
       strideY, strideX,
       padTop, padBottom,
       padLeft, padRight) ;
      break;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return status ;
}


/* ---------------------------------------------------------------- */
/*                                                  nnconvt_forward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnconvt_forward(Context& context,
                    Tensor output,
                    Tensor data,
                    Tensor filters,
                    Tensor biases,
                    int upsampleY, int upsampleX,
                    int cropTop, int cropBottom,
                    int cropLeft, int cropRight)
{
  vl::Error status = vlSuccess ;
  size_t dataOffset = data.getHeight()*data.getWidth()*data.getDepth() ;
  size_t outputOffset = output.getHeight()*output.getWidth()*output.getDepth() ;

  // we need to process this down per image as nnconv_backward would otherwise
  // accumulate everything into a single feature field in output
  for (int image = 0 ; image < data.getSize() ; ++image) {
    Tensor dataSlice(data) ;
    dataSlice.setMemory(data.getMemory() + dataOffset * image) ;
    dataSlice.setSize(1) ;

    Tensor outputSlice(output) ;
    outputSlice.setMemory(output.getMemory() + outputOffset * image) ;
    outputSlice.setSize(1) ;

    status = vl::nnconv_backward(context,
                                 outputSlice, Tensor(), Tensor(),
                                 Tensor(), filters, dataSlice,
                                 upsampleY, upsampleX,
                                 cropTop, cropBottom,
                                 cropLeft, cropRight) ;
    if (status != vlSuccess) { goto done ; }
  }
  if (biases) {
    status = vl::nnbias_forward(context,
                                output, 1,
                                Tensor(), 0,
                                biases, 1) ;
  }
done:
  return status ;
}

/* ---------------------------------------------------------------- */
/*                                                 nnconvt_backward */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnconvt_backward(Context& context,
                     Tensor derData,
                     Tensor derFilters,
                     Tensor derBiases,
                     Tensor data,
                     Tensor filters,
                     Tensor derOutput,
                     int upsampleY, int upsampleX,
                     int cropTop, int cropBottom,
                     int cropLeft, int cropRight)
{
  vl::Error status = vl::vlSuccess ;

  if (derData) {
    status = vl::nnconv_forward(context,
                                derData, 0,
                                derOutput, 1,
                                filters, Tensor(),
                                upsampleY, upsampleX,
                                cropTop, cropBottom,
                                cropLeft, cropRight) ;
    if (status != vlSuccess) { goto done ; }
  }

  if (derFilters) {
    status = vl::nnconv_backward(context,
                                 Tensor(), derFilters, Tensor(),
                                 derOutput, Tensor(), data,
                                 upsampleY, upsampleX,
                                 cropTop, cropBottom,
                                 cropLeft, cropRight) ;
    if (status != vlSuccess) { goto done ; }
  }

  if (derBiases) {
    status = vl::nnbias_backward(context,
                                 Tensor(), 0,
                                 derBiases, 0,
                                 derOutput, 1) ;
  }

done:
  return status ;
}
