#include "bilinearsampler.hpp"
#include "../data.hpp"
#include <assert.h>
#include <float.h>
#include <cstdio>
#include <math.h>
#include <string.h>

// use a template to define both given similarities
template<typename type, bool backward>
static vl::Error
forward_backward_data
(vl::Context& context,
 type* output, // null in backward
 type* derData, // null in forward
 type const* data, // null in backward
 type const* grid,
 type const* derOutput, // null in forward
 size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
 size_t inHeight, size_t inWidth, size_t inCardinality)
{
  vl::Error error = vl::vlSuccess ;

  assert(grid) ;
  assert(divides(inCardinality, outCardinality)) ;

  // forward conditions
  assert(backward || data) ;
  assert(backward || output) ;
  assert(backward || derData == NULL) ;
  assert(backward || derOutput == NULL) ;

  // backward conditions
  assert(!backward || data == NULL) ;
  assert(!backward || output == NULL) ;
  assert(!backward || derData) ;
  assert(!backward || derOutput) ;

  int groupSize = outCardinality / inCardinality ;

  if (backward) {
    memset(derData, 0, inHeight * inWidth * outDepth * inCardinality * sizeof(type)) ;
  }

  for (int n = 0 ; n < outCardinality ; ++n) {
    for (int c = 0 ; c < outDepth ; ++c) {
      type const * end = grid + 2 * outWidth * outHeight ;
      while (grid < end) {
        type py = *grid++ ;
        type px = *grid++ ;

        py = (py + 1.0) / 2.0 * (inHeight - 1) ;
        px = (px + 1.0) / 2.0 * (inWidth - 1) ;
        const int sx = floor(px);
        const int sy = floor(py);

        type dy ;
        if (backward) {
          dy = *derOutput++ ;
        }

        // skip if out of range
        if (sy < -1 || sy > inHeight - 1 || sx < -1 || sx > inWidth - 1) {
          if (!backward) {
            *output++ = 0 ;
          }
          continue ;
        }

        // get the interpolation weights
        const type wx = px - sx ;
        const type wy = py - sy ;

        // add the weighted sum to the output:
        int ssx, ssy;
        type acc = 0;

        // get the number of input-image from which we get the data:
        // this is NOT always the same as the affine-grid image number
        // as there can be multiple GRIDS per input image:
        for (int j=0; j< 2; j++) {
          for (int i=0; i < 2; i++) {
            ssy = sy + i;
            ssx = sx + j;
            if (ssy < 0 || ssy > inHeight - 1 || ssx < 0 || ssx > inWidth - 1) {
              continue ;
            }
            const type w = ((1-j)*(1-wx) + j*wx) * ((1-i)*(1-wy) + i*wy);
            if (!backward) {
              acc += w * data[ssy + ssx * inHeight];
            } else {
              derData[ssy + ssx * inHeight] += w * dy ;
            }
          }
        }
        if (!backward) {
          *output++ = acc ;
        }
      }
      // next channel
      if (!backward) {
        data += inHeight * inWidth ;
      } else {
        derData +=inHeight * inWidth ;
      }
      grid -= 2 * outHeight * outWidth ;
    }
    // next image
    if ((n + 1) % groupSize != 0) {
      data -= inHeight * inWidth * outDepth ;
      derData -= inHeight * inWidth * outDepth ;
    }
    grid += 2 * outHeight * outWidth ;
  }
  return error ;
}

namespace vl { namespace impl {

  template<typename type>
  struct bilinearsampler<vl::CPU, type>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::Error
    forward(Context& context,
            type* output,
            type const* data,
            type const* grid,
            size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
            size_t inHeight, size_t inWidth, size_t inCardinality)
    {
      return forward_backward_data<type, false>
      (context, output, NULL, data, grid, NULL,
       outHeight, outWidth, outDepth, outCardinality,
       inHeight, inWidth,inCardinality) ;
    }


    /*------------------------------------------------------------- */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::Error
    backward(Context& context,
             type* derData,
             type* derGrid,
             type const* data,
             type const* grid,
             type const* derOutput,
             size_t outHeight, size_t outWidth, size_t outDepth, size_t outCardinality,
             size_t inHeight, size_t inWidth, size_t inCardinality)
    {
      vl::Error error = vlSuccess ;

      error = forward_backward_data<type, true>
      (context, NULL, derData, NULL, grid, derOutput,
       outHeight, outWidth, outDepth, outCardinality,
       inHeight, inWidth,inCardinality) ;
      if (error != vlSuccess) { return error; }

      // todo: backward grid
      return error ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::bilinearsampler<vl::CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bilinearsampler<vl::CPU, double> ;
#endif
