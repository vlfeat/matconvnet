#include "bilinearsampler.hpp"
#include "../data.hpp"
#include <assert.h>
#include <float.h>
#include <cstdio>
#include <math.h>
#include <string.h>

// use a template to define both directions as they are nearly identical code-wise
template<typename type, bool backwardData, bool backwardGrid>
static vl::Error
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
  vl::Error error = vl::vlSuccess ;

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

  if (backwardData) {
    memset(derData, 0, inHeight * inWidth * outDepth * inCardinality * sizeof(type)) ;
  }

  for (int n = 0 ; n < outCardinality ; ++n) {
    for (int c = 0 ; c < outDepth ; ++c) {
      type const * end = grid + 2 * outWidth * outHeight ;
      derGrid -= 2 ;
      while (grid < end) {
        type py = *grid++ ;
        type px = *grid++ ;
        derGrid += 2 ;

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

        // add the weighted sum to the output
        type acc = 0;

        #pragma unroll
        for (int j=0; j < 2; j++) {
          #pragma unroll
          for (int i=0; i < 2; i++) {
            int ssy = sy + i ;
            int ssx = sx + j ;
            if (ssy < 0 || ssy > inHeight - 1 || ssx < 0 || ssx > inWidth - 1) {
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
                derGrid[0] += wwy * dy ;
                derGrid[1] += wwx * dy ;
              }
            }
          }
        }
        if (!backward) {
          *output++ = acc ;
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
      return forward_backward<type, false, false>
      (context, output, NULL, NULL, data, grid, NULL,
       outHeight, outWidth, outDepth, outCardinality,
       inHeight, inWidth,inCardinality) ;
    }


    /*------------------------------------------------------------- */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

#define DISPATCH(bwData, bwGrid) \
error = forward_backward<type, bwData, bwGrid> \
    (context, NULL, derData, derGrid, data, grid, derOutput, \
     outHeight, outWidth, outDepth, outCardinality, \
     inHeight, inWidth,inCardinality) ;

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

      // optimized codepaths depending on what needs to be comptued
      if (derData && derGrid == NULL) {
        DISPATCH(true, false) ;
      } else if (derGrid && derData == NULL) {
        DISPATCH(false, true) ;
      } else if (derGrid && derData) {
        DISPATCH(true, true) ;
      }
      return error ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::bilinearsampler<vl::CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bilinearsampler<vl::CPU, double> ;
#endif
