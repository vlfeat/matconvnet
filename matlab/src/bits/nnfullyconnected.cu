//
//  nnfullyconnected.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 05/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "nnfullyconnected.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* nnfullyconnected_forward_impl                                    */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type>
int nnfullyconnected_forward_impl(Context& context,
                                  Tensor output,
                                  Tensor data,
                                  Tensor filters,
                                  Tensor biases)
{
  float alpha = 1 ;
  float beta = 0 ;

  if (filters) {
    ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;
    if (data.getSize() == 1) {
      /* one image in the stack */
      gemv<arch,type>(context,
                      't',
                      filtersVolume, filters.getSize(),
                      alpha,
                      (type*)filters.getMemory(), filtersVolume,
                      (type*)data.getMemory(), 1,
                      beta,
                      (type*)output.getMemory(), 1) ;
    } else {
      /* multiple images in the stack */
      gemm<arch,type>(context,
                      't', 'n',
                      filters.getSize(), data.getSize(), filtersVolume,
                      alpha,
                      (type*)filters.getMemory(), filtersVolume,
                      (type*)data.getMemory(), filtersVolume,
                      beta,
                      (type*)output.getMemory(), filters.getSize()) ;
    }
  } else {
    vl::impl::copy<arch,type>(output.getMemory(),
                              data.getMemory(),
                              data.getNumElements()) ;
  }

  if (biases) {
    float beta = 1 ;
    type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                           get_type_id<type>(),
                                                           data.getSize()) ;
    gemm<arch,type>(context, 'n', 'n',
                    biases.getNumElements(), data.getSize(), 1,
                    alpha,
                    (type*)biases.getMemory(), biases.getNumElements(),
                    allOnesMemory, 1,
                    beta,
                    (type*)output.getMemory(), biases.getNumElements()) ;
  }
  return 0 ;
}

/* ---------------------------------------------------------------- */
/* nnfullyconnected_backward_impl                                   */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type>
int nnfullyconnected_backward_impl(vl::Context& context,
                                   vl::Tensor derData,
                                   vl::Tensor derFilters,
                                   vl::Tensor derBiases,
                            vl::Tensor data,
                                   vl::Tensor filters,
                                   vl::Tensor derOutput)
{
  float alpha = 1 ;
  float beta = 0 ;

  if (filters) {
    ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;

    if (derFilters) {
      gemm<arch, type>(context,
                       'n', 't',
                       filtersVolume, filters.getSize(), data.getSize(),
                       alpha,
                       (type*)data.getMemory(), filtersVolume,
                       (type*)derOutput.getMemory(), filters.getSize(),
                       beta,
                       (type*)derFilters.getMemory(), filtersVolume) ;
    }

    if (derData) {
      gemm<arch, type>(context,
                       'n', 'n',
                       filtersVolume, data.getSize(), filters.getSize(),
                       alpha,
                       (type*)filters.getMemory(), filtersVolume,
                       (type*)derOutput.getMemory(), filters.getSize(),
                       beta,
                       (type*)derData.getMemory(), filtersVolume) ;
    }
  } else {
    vl::impl::copy<arch,type>(derData.getMemory(),
                              derOutput.getMemory(),
                              derOutput.getNumElements()) ;
  }

  if (derBiases) {
    type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                           get_type_id<type>(),
                                                           derOutput.getSize()) ;

    gemm<arch, type>(context,
                     'n', 't',
                     1, derOutput.getDepth(), derOutput.getSize(),
                     alpha,
                     (type*)allOnesMemory, 1,
                     (type*)derOutput.getMemory(), derOutput.getDepth(),
                     beta,
                     (type*)derBiases.getMemory(), 1) ;
  }
  return 0 ;
}

/* ---------------------------------------------------------------- */
/* nnfullyconnected_forward                                         */
/* ---------------------------------------------------------------- */

int vl::nnfullyconnected_forward(Context& context,
                                 Tensor output,
                                 Tensor data,
                                 Tensor filters,
                                 Tensor biases)
{
  int status = 0 ;
  switch (data.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::ERROR ;
      break ;

    case vl::CPU:
      status = nnfullyconnected_forward_impl<CPU,float>
      (context, output, data, filters, biases) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      status = nnfullyconnected_forward_impl<GPU,float>
      (context, output, data, filters, biases) ;
      break ;
#endif
  }
  return status ;
}


/* ---------------------------------------------------------------- */
/* nnfullyconnected_backward                                        */
/* ---------------------------------------------------------------- */

int vl::nnfullyconnected_backward(vl::Context& context,
                             vl::Tensor derData,
                                  vl::Tensor derFilters,
                                  vl::Tensor derBiases,
                             vl::Tensor data,
                                  vl::Tensor filters,
                                  vl::Tensor derOutput)
{
  int status = 0 ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::ERROR ;
      break ;

    case vl::CPU:
      status = nnfullyconnected_backward_impl<CPU,float>
      (context, derData, derFilters, derBiases, data, filters, derOutput) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      status = nnfullyconnected_backward_impl<GPU,float>
      (context, derData, derFilters, derBiases, data, filters, derOutput) ;
      break ;
#endif
  }
  return status ;
}


