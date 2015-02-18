// @file nnconv_blas.hpp
// @brief Convolution block BLAS-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv_blas__
#define __vl__nnconv_blas__

#include "im2row.hpp"
#include "blashelper.hpp"
#include <assert.h>

namespace vl { namespace impl {

  template<vl::Device arch, typename type> inline vl::Error
  nnconv_forward_blas(Context& context,
                      Tensor output,
                      Tensor data,
                      Tensor filters,
                      Tensor biases,
                      int strideY, int strideX,
                      int padTop, int padBottom,
                      int padLeft, int padRight) ;

  template<vl::Device arch, typename type> inline vl::Error
  nnconv_backward_blas(Context& context,
                       Tensor derData,
                       Tensor derFilters,
                       Tensor derBiases,
                       Tensor data,
                       Tensor filters,
                       Tensor derOutput,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight) ;

} }

/*

 One image at a time is processed.

 Filters are (optionally) divided in to groups, one for each group of dimensions.


                 patchVolume                  numFilters
                 +-------------------------+   +-----------------------+

                 filtersVolume              numFiltersPerGroup
                 +------------+------------+   +-----------+-----------+      +--------+--------+
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |  filter   |           |      |        |        |
                 |            |            |   |  group 1  |     0     |  =   |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------------------+      |        |        |
 numOutputPixels |   grp. 1   |   grp. 2   |   |           |           |      |        |        |
                 |            |            |   |           |  filter   |      |        |        |
                 |            |            |   |     0     |  group 2  |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------+-----------+      |        |        |
                 |            |            |                                  |        |        |
                 |            |            |            filters               |        |        |
                 |            |            |                                  |        |        |
                 +------------+------------+                                  +--------+--------+

                 temp                                                     output

 */

template<vl::Device arch, typename type> inline vl::Error
vl::impl::nnconv_forward_blas(Context& context,
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

  vl::Error error ;

  ptrdiff_t numGroups = data.getDepth() / filters.getDepth() ;
  ptrdiff_t numFiltersPerGroup = filters.getSize() / numGroups ;
  ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
  ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;
  ptrdiff_t tempVolume = numOutputPixels * filtersVolume * numGroups ;

  type* tempMemory = (type*) context.getWorkspace(arch, tempVolume * sizeof(type)) ;
  type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                         get_vl_type<type>(),
                                                         numOutputPixels) ;
  if (tempMemory == NULL || allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < data.getSize() ; ++image) {

    ptrdiff_t dataOffset = (data.getHeight()*data.getWidth()*data.getDepth()) * image ;
    ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;

    error = vl::impl::im2row<arch,type>(context,
                                        tempMemory,
                                        (type*)data.getMemory() + dataOffset,
                                        data.getHeight(), data.getWidth(), data.getDepth(),
                                        filters.getHeight(), filters.getWidth(),
                                        strideY, strideX,
                                        padTop, padBottom, padLeft, padRight) ;
    if (error != vl::vlSuccess) { goto done ; }

    for (int g = 0 ; g < numGroups ; ++ g) {
      ptrdiff_t filterGrpOffset = filtersVolume * numFiltersPerGroup * g ;
      ptrdiff_t tempGrpOffset = numOutputPixels * filtersVolume * g ;
      ptrdiff_t outputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
      type alpha = 1 ;
      type beta = 0 ;
      error = gemm<arch,type>(context,
                              'n', 'n',
                              numOutputPixels, numFiltersPerGroup, filtersVolume,
                              alpha,
                              tempMemory + tempGrpOffset, numOutputPixels,
                              (type*)filters.getMemory() + filterGrpOffset, filtersVolume,
                              beta,
                              (type*)output.getMemory() + outputOffset + outputGrpOffset, numOutputPixels) ;
      if (error != vl::vlSuccess) { goto done ; }
    }

    if (biases) {
      type alpha = 1 ;
      type beta = 1 ;
      error = gemm<arch,type>(context,
                              'n', 'n',
                              numOutputPixels, biases.getNumElements(), 1,
                              alpha,
                              allOnesMemory, numOutputPixels,
                              (type*)biases.getMemory(), 1,
                              beta,
                              (type*)output.getMemory() + outputOffset, numOutputPixels) ;
      if (error != vl::vlSuccess) { goto done ; }
    }
  }

done:
  return context.passError(error, "nnconv_forward_blas<>: ") ;
}

template<vl::Device arch, typename type> inline vl::Error
vl::impl::nnconv_backward_blas(Context& context,
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

  vl::Error error ;

  ptrdiff_t numGroups = data.getDepth() / filters.getDepth() ;
  ptrdiff_t numFiltersPerGroup = filters.getSize() / numGroups ;
  ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
  ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;
  ptrdiff_t tempVolume = numOutputPixels * filtersVolume * numGroups ;

  type* tempMemory = (type*) context.getWorkspace(arch, tempVolume * sizeof(type)) ;
  type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                         get_vl_type<type>(),
                                                         numOutputPixels) ;
  if (tempMemory == NULL || allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < data.getSize() ; ++image) {

    ptrdiff_t derDataOffset = (data.getHeight()*data.getWidth()*data.getDepth()) * image ;
    ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

    /* compute derFilters dz/dF */
    if (derFilters) {
      error = vl::impl::im2row<arch,type>(context,
                                          (type*)tempMemory,
                                          (type*)data.getMemory() + derDataOffset,
                                          data.getHeight(), data.getWidth(), data.getDepth(),
                                          filters.getHeight(), filters.getWidth(),
                                          strideY, strideX,
                                          padTop, padBottom, padLeft, padRight) ;
      if (error != vl::vlSuccess) { return error ; }
      for (int g = 0 ; g < numGroups ; ++ g) {
        ptrdiff_t filterGrpOffset = filtersVolume * numFiltersPerGroup * g ;
        ptrdiff_t tempGrpOffset = numOutputPixels * filtersVolume * g ;
        ptrdiff_t derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        /* dzdF = temp' * dzdY */
        type alpha = 1 ;
        type beta = (image > 0) ; /* this saves init. the output array with 0 */
        error = gemm<arch,type>(context,
                                't', 'n',
                                filtersVolume, numFiltersPerGroup, numOutputPixels,
                                alpha,
                                tempMemory + tempGrpOffset, numOutputPixels,
                                (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
                                beta,
                                (type*)derFilters.getMemory() + filterGrpOffset, filtersVolume) ;
        if (error != vl::vlSuccess) { return error ; }
      }
    }

    /* compute derData dz/dbias */
    if (derBiases) {
      type alpha = 1 ;
      type beta = (image > 0) ; /* this saves init. the output array with 0 */
      error = gemv<arch,type>(context,
                              't',
                              numOutputPixels, filters.getSize(),
                              alpha, /* alpha */
                              derOutput.getMemory() + derOutputOffset, numOutputPixels,
                              allOnesMemory, 1,
                              beta, /* beta */
                              derBiases.getMemory(), 1) ;
      if (error != vl::vlSuccess) { return error ; }
    }

    /* compute derData dz/dx */
    if (derData) {
      for (int g = 0 ; g < numGroups ; ++ g) {
        ptrdiff_t filterGrpOffset = filtersVolume * numFiltersPerGroup * g ;
        ptrdiff_t tempGrpOffset = numOutputPixels * filtersVolume * g ;
        ptrdiff_t derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        float alpha = 1 ;
        float beta = 0 ;
        error = gemm<arch,type>(context,
                                'n', 't',
                                numOutputPixels, filtersVolume, numFiltersPerGroup,
                                alpha,
                                (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
                                (type*)filters.getMemory() + filterGrpOffset, filtersVolume,
                                beta,
                                tempMemory + tempGrpOffset, numOutputPixels) ;
        if (error != vl::vlSuccess) { return error ; }
      }
      error = vl::impl::row2im<arch,type>(context,
                                          (type*)derData.getMemory() + derDataOffset,
                                          tempMemory,
                                          data.getHeight(), data.getWidth(), data.getDepth(),
                                          filters.getHeight(), filters.getWidth(),
                                          strideY, strideX,
                                          padTop, padBottom, padLeft, padRight) ;
      if (error != vl::vlSuccess) { return error ; }
    }
  }

done:
  return context.passError(error, "nnconv_backward_blas<>: ") ;
}

#endif /* defined(__vl__nnconv_blas__) */
