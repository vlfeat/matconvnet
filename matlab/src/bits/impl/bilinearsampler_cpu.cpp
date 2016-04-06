// defines the CUDA kernels and the dispatcher for them:

#include "bilinearsampler.hpp"
#include "../data.hpp"
#include <assert.h>
#include <float.h>
#include <cstdio>
#include <math.h>

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
      vl::Error error = vlSuccess ;

      assert(output) ;
      assert(data) ;
      assert(grid) ;
      assert(divides(inCardinality, outCardinality)) ;
      int groupSize = outCardinality / inCardinality ;

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

            // skip if out of range
            if (sy < -1 || sy > inHeight - 1 || sx < -1 || sx > inWidth - 1) {
              *output++ = 0;
              continue ;
            }

            // get the interpolation weights
            const type wx = px - sx ;
            const type wy = py - sy ;

            // add the weighted sum to the output:
            int ssx, ssy;
            type outval = 0;

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
                outval += w * data[ssy + ssx * inHeight];
              }
            }
            *output++ = outval ;
          }
          // next channel
          data += inHeight * inWidth ;
          grid -= 2 * outHeight * outWidth ;
        }
        // next image
        if ((n + 1) % groupSize != 0) {
          data -= inHeight * inWidth * outDepth ;
        }
        grid += 2 * outHeight * outWidth ;
      }
      return error ;
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
      return error ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::bilinearsampler<vl::CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bilinearsampler<vl::CPU, double> ;
#endif
