#ifndef VL_BILINEARSAMPLER_H
#define VL_BILINEARSAMPLER_H

#include "../data.hpp"
#include <cstddef>

// defines the dispatcher for CUDA kernels:
namespace vl { namespace impl {

  template<vl::Device dev, typename type>
  struct bilinearsampler {

    static vl::Error
    forward(Context& context,
            type* output,
            type const* data,
            type const* grid,
            size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
            size_t inHeight, size_t inWidth, size_t inCardinality) ;


    static vl::Error
    backward(Context& context,
             type* derData,
             type* derGrid,
             type const* data,
             type const* grid,
             type const* derOutput,
             size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
             size_t inHeight, size_t inWidth, size_t inCardinality) ;
  } ;

} }

#endif /* defined(VL_BILINEARSAMPLER_H) */
