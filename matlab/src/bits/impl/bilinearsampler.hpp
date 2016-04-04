#ifndef VL_BILINEARSAMPLER_H
#define VL_BILINEARSAMPLER_H

#include "../data.hpp"
#include <cstddef>

// defines the dispatcher for CUDA kernels:
namespace vl { namespace impl {

  struct bilinearsampler {

    static vl::Error
    forward(float* output,
            float const* data,
            float const* grid,
            size_t outHeight, size_t outWidth,
            size_t nBatch_grid,
            size_t inHeight, size_t inWidth,
            size_t nChannels, size_t nBatch_data) ;            


    static vl::Error
    backward(float* derData,
             float* derGrid,
             float const* data,
             float const* grid,
             float const* derOutput,
             size_t outHeight, size_t outWidth,
             size_t nBatch_grid,
             size_t inHeight, size_t inWidth,
             size_t nChannels,
             size_t nBatch_data) ;
  } ;
} }

#endif /* defined(VL_POOLING_H) */
