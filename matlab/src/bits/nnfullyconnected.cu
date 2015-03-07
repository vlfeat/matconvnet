// @file nnfullyconnected.cu
// @brief Fully-connected block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnfullyconnected.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* nnfullyconnected_forward_impl                                    */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type> vl::Error
nnfullyconnected_forward_impl(Context& context,
                              Tensor output,
                              Tensor data,
                              Tensor filters,
                              Tensor biases)
{
  float alpha = 1 ;
  float beta = 0 ;

  vl::Error error ;

  if (filters) {
    ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;
    if (data.getSize() == 1) {
      /* one image in the stack */
      error = gemv<arch,type>(context,
                              't',
                              filtersVolume, filters.getSize(),
                              alpha,
                              (type*)filters.getMemory(), filtersVolume,
                              (type*)data.getMemory(), 1,
                              beta,
                              (type*)output.getMemory(), 1) ;
      if (error != vl::vlSuccess) { goto done ; }
    } else {
      /* multiple images in the stack */
      error = gemm<arch,type>(context,
                              't', 'n',
                              filters.getSize(), data.getSize(), filtersVolume,
                              alpha,
                              (type*)filters.getMemory(), filtersVolume,
                              (type*)data.getMemory(), filtersVolume,
                              beta,
                              (type*)output.getMemory(), filters.getSize()) ;
      if (error != vl::vlSuccess) { goto done ; }
    }
  } else {
    error = vl::impl::copy<arch,type>(output.getMemory(),
                                      data.getMemory(),
                                      data.getNumElements()) ;
  }

  if (biases) {
    float beta = 1 ;
    type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                           get_vl_type<type>(),
                                                           data.getSize()) ;
    if (allOnesMemory == NULL) {
      error = context.getLastError() ;
      goto done ;
    }
    error = gemm<arch,type>(context, 'n', 'n',
                            biases.getNumElements(), data.getSize(), 1,
                            alpha,
                            (type*)biases.getMemory(), biases.getNumElements(),
                            allOnesMemory, 1,
                            beta,
                            (type*)output.getMemory(), biases.getNumElements()) ;
    if (error != vl::vlSuccess) { goto done ; }
  }
done:
  return context.passError(error, "nnfullyconnected_forward_impl<>: ") ;
}

/* ---------------------------------------------------------------- */
/* nnfullyconnected_backward_impl                                   */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type> vl::Error
nnfullyconnected_backward_impl(vl::Context& context,
                               vl::Tensor derData,
                               vl::Tensor derFilters,
                               vl::Tensor derBiases,
                               vl::Tensor data,
                               vl::Tensor filters,
                               vl::Tensor derOutput)
{
  float alpha = 1 ;
  float beta = 0 ;

  vl::Error error ;

  if (filters) {
    ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;

    if (derFilters) {
      error = gemm<arch, type>(context,
                               'n', 't',
                               filtersVolume, filters.getSize(), data.getSize(),
                               alpha,
                               (type*)data.getMemory(), filtersVolume,
                               (type*)derOutput.getMemory(), filters.getSize(),
                               beta,
                               (type*)derFilters.getMemory(), filtersVolume) ;
      if (error != vl::vlSuccess) { goto done ; }
    }

    if (derData) {
      error = gemm<arch, type>(context,
                       'n', 'n',
                       filtersVolume, data.getSize(), filters.getSize(),
                       alpha,
                       (type*)filters.getMemory(), filtersVolume,
                       (type*)derOutput.getMemory(), filters.getSize(),
                       beta,
                       (type*)derData.getMemory(), filtersVolume) ;
      if (error != vl::vlSuccess) { goto done ; }
    }
  } else {
    vl::impl::copy<arch,type>(derData.getMemory(),
                              derOutput.getMemory(),
                              derOutput.getNumElements()) ;
  }

  if (derBiases) {
    type const* allOnesMemory = (type*) context.getAllOnes(arch,
                                                           get_vl_type<type>(),
                                                           derOutput.getSize()) ;
    if (allOnesMemory == NULL) {
      error = context.getLastError() ;
      goto done ;
    }

    error = gemm<arch, type>(context,
                     'n', 't',
                     1, derOutput.getDepth(), derOutput.getSize(),
                     alpha,
                     (type*)allOnesMemory, 1,
                     (type*)derOutput.getMemory(), derOutput.getDepth(),
                     beta,
                     (type*)derBiases.getMemory(), 1) ;
    if (error != vl::vlSuccess) { goto done ; }

  }
done:
  return context.passError(error, "nnfullyconnected_backward_impl<>: ") ;
}

/* ---------------------------------------------------------------- */
/* nnfullyconnected_forward                                         */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnfullyconnected_forward(Context& context,
                                 Tensor output,
                                 Tensor data,
                                 Tensor filters,
                                 Tensor biases)
{
  vl::Error status = vl::vlSuccess ;
  switch (data.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
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
  return context.passError(status, "nnfullyconnected_forward") ;
}


/* ---------------------------------------------------------------- */
/* nnfullyconnected_backward                                        */
/* ---------------------------------------------------------------- */

vl::Error
vl::nnfullyconnected_backward(vl::Context& context,
                             vl::Tensor derData,
                                  vl::Tensor derFilters,
                                  vl::Tensor derBiases,
                             vl::Tensor data,
                                  vl::Tensor filters,
                                  vl::Tensor derOutput)
{
  vl::Error status = vl::vlSuccess ;
  switch (derOutput.getMemoryType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
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
  return context.passError(status, "nnfullyconnected_backward") ;
}


