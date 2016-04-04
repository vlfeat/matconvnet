// @file nnbilinearsampler.cu
// @brief Bilinear Sampler block
// @author Ankush Gupta

/*
Copyright (C) 2016 Ankush Gupta.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbilinearsampler.hpp"
#include "impl/bilinearsampler.hpp"

#include <cstdio>

//#if ENABLE_GPU
#include "datacu.hpp"
//#endif

#include <assert.h>

using namespace vl ;


vl::Error
vl::nnbilinearsampler_forward(Context& context,
                              Tensor output,
                              Tensor data,
                              Tensor grid) {

  vl::Error status = vlSuccess ;
  vl::Device deviceType = output.getDeviceType() ;
  vl::Type dataType = output.getDataType() ;

  switch (deviceType) {
    default:
      assert(false);
      return vl::vlErrorUnknown;

    case vl::CPU:
      assert(false);
      return vl::vlErrorUnsupported;

    case vl::GPU:

      status = vl::impl::bilinearsampler::forward((float*) output.getMemory(),
                                        (float const*) data.getMemory(),
                                        (float const*) grid.getMemory(),
                                        output.getHeight(), output.getWidth(),
                                        output.getSize(),
                                        data.getHeight(), data.getWidth(),
                                        data.getDepth(), data.getSize());
      if (status==vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__));
      }
      break;
  }

  return context.passError(status, "nnbilinearsampler_forward");
}

vl::Error
vl::nnbilinearsampler_backward( Context& context,
                                Tensor derData,
                                Tensor derGrid,
                                Tensor data,
                                Tensor grid,
                                Tensor derOutput)
{
  vl::Error status = vlSuccess ;
  vl::Device deviceType = derOutput.getDeviceType() ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false);
      return vl::vlErrorUnknown;

    case vl::CPU:
      assert(false);
      return vl::vlErrorUnsupported;

    case vl::GPU:
      status = vl::impl::bilinearsampler::backward((float*) derData.getMemory(),
                                         (float *) derGrid.getMemory(),
                                         (float const*) data.getMemory(),
                                         (float const*) grid.getMemory(),
                                         (float const*) derOutput.getMemory(),
                                         derOutput.getHeight(), derOutput.getWidth(),
                                         derOutput.getSize(),
                                         data.getHeight(), data.getWidth(),
                                         data.getDepth(), data.getSize());
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("bilinearsampler_*::backward"));
      }
      break;
  }
  return context.passError(status, "nnbilinearsampler_backward");
}


