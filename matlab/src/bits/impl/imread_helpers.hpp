// @file imread_helpers.cpp
// @brief Image reader helper functions.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <algorithm>
#include <cassert>

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

namespace vl { namespace impl {

  enum pixelFormatId {
    pixelFormatL,
    pixelFormatRGB,
    pixelFormatRGBA,
    pixelFormatBGR,
    pixelFormatBGRA,
	pixelFormatBGRAasL
  };

#ifndef __SSSE3__
#ifdef _MSC_VER
#pragma message ( "SSSE3 instruction set not enabled. Using slower image conversion routines." )
#else
#warning "SSSE3 instruction set not enabled. Using slower image conversion routines."
#endif

  template<int pixelFormat> void
  imageFromPixels(vl::Image & image, char unsigned const * rgb, int rowStride)
  {
    int blockSizeX ;
    int blockSizeY ;
    int pixelStride ;
    int imagePlaneStride = image.width * image.height ;
    switch (pixelFormat) {
      case pixelFormatL:
        pixelStride = 1 ;
        blockSizeX = 16 ;
        blockSizeY = 4 ;
        break ;
      case pixelFormatBGR:
      case pixelFormatRGB:
        pixelStride = 3 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        break ;
      case pixelFormatRGBA:
      case pixelFormatBGRA:
	  case pixelFormatBGRAasL:
        pixelStride = 4 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        break ;
      default:
        assert(false) ;
    }

    // we pull out these values as otherwise the compiler
    // will assume that the reference &image can be aliased
    // and recompute silly multiplications in the inner loop

    float * const  __restrict imageMemory = image.memory ;
    int const imageHeight = image.height ;
    int const imageWidth = image.width ;

    for (int x = 0 ; x < imageWidth ; x += blockSizeX) {
      float * __restrict imageMemoryX = imageMemory + x * imageHeight ;
      int bsx = (std::min)(imageWidth - x, blockSizeX) ;

      for (int y = 0 ; y < imageHeight ; y += blockSizeY) {
        int bsy = (std::min)(imageHeight - y, blockSizeY) ;
        float * __restrict r ;
        float * rend ;
        for (int dx = 0 ; dx < bsx ; ++dx) {
          char unsigned const * __restrict pixel = rgb + y * rowStride + (x + dx) * pixelStride ;
          r = imageMemoryX + y + dx * imageHeight ;
          rend = r + bsy ;
          while (r != rend) {
            switch (pixelFormat) {
              case pixelFormatRGBA:
              case pixelFormatRGB:
                r[0 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[2 * imagePlaneStride] = (float) pixel[2] ;
                break ;
              case pixelFormatBGR:
              case pixelFormatBGRA:
                r[2 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[0 * imagePlaneStride] = (float) pixel[2] ;
                break;
			  case pixelFormatBGRAasL:
              case pixelFormatL:
                r[0] = (float) pixel[0] ;
                break ;
            }
            r += 1 ;
            pixel += rowStride ;
          }
        }
      }
    }
  }

#else
#ifdef _MSC_VER
  #pragma message ( "SSSE3 instruction set enabled." )
#endif
  /* SSSE3 optimised version */

  template<int pixelFormat> void
  imageFromPixels(vl::Image & image, char unsigned const * rgb, int rowStride)
  {
    int blockSizeX ;
    int blockSizeY ;
    int pixelStride ;
    int imagePlaneStride = image.width * image.height ;
    __m128i shuffleRgb ;
    __m128i const shuffleL = _mm_set_epi8(0xff, 0xff, 0xff,  3,
                                          0xff, 0xff, 0xff,  2,
                                          0xff, 0xff, 0xff,  1,
                                          0xff, 0xff, 0xff,  0) ;
    __m128i const mask = _mm_set_epi32(0xff, 0xff, 0xff, 0xff) ;

    switch (pixelFormat) {
      case pixelFormatL:
        pixelStride = 1 ;
        blockSizeX = 16 ;
        blockSizeY = 4 ;
        break ;
      case pixelFormatBGR:
      case pixelFormatRGB:
        pixelStride = 3 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        assert(image.depth == 3) ;
        break ;
      case pixelFormatRGBA:
	  case pixelFormatBGRA:
	  case pixelFormatBGRAasL:
        pixelStride = 4 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        assert(image.depth == 3) ;
        break ;
      default:
        assert(false) ;
    }

    switch (pixelFormat) {
      case pixelFormatL:
        break ;

      case pixelFormatRGB:
        shuffleRgb = _mm_set_epi8(0xff, 11, 10,  9,
                                  0xff,  8,  7,  6,
                                  0xff,  5,  4,  3,
                                  0xff,  2,  1,  0) ;
        break ;

      case pixelFormatRGBA:
        shuffleRgb = _mm_set_epi8(0xff, 14, 13, 12,
                                  0xff, 10,  9,  8,
                                  0xff,  6,  5,  4,
                                  0xff,  2,  1,  0) ;
        break ;

      case pixelFormatBGR:
        shuffleRgb = _mm_set_epi8(0xff,  9, 10, 11,
                                  0xff,  6,  7,  8,
                                  0xff,  3,  4,  4,
                                  0xff,  0,  1,  2) ;
        break ;

      case pixelFormatBGRA:
        shuffleRgb = _mm_set_epi8(0xff, 12, 13, 14,
                                  0xff,  8,  9, 10,
                                  0xff,  4,  5,  6,
                                  0xff,  0,  1,  2) ;
        break ;

	  case pixelFormatBGRAasL:
        shuffleRgb = _mm_set_epi8(0xff, 0xff, 0xff, 12,
                                  0xff, 0xff, 0xff, 8,
                                  0xff, 0xff, 0xff, 4,
                                  0xff, 0xff, 0xff, 0) ;
        break ;
    }

    // we pull out these values as otherwise the compiler
    // will assume that the reference &image can be aliased
    // and recompute silly multiplications in the inner loop
    float *  const __restrict imageMemory = image.memory ;
    int const imageHeight = image.height ;
    int const imageWidth = image.width ;

    for (int x = 0 ; x < imageWidth ; x += blockSizeX) {
      int y = 0 ;
      float * __restrict imageMemoryX = imageMemory + x * imageHeight ;
      int bsx = (std::min)(imageWidth - x, blockSizeX) ;
      if (bsx < blockSizeX) goto boundary ;

      for ( ; y < imageHeight - blockSizeY + 1 ; y += blockSizeY) {
        char unsigned const * __restrict pixel = rgb + y * rowStride + x * pixelStride ;
        float * __restrict r = imageMemoryX + y ;
        __m128i p0, p1, p2, p3, T0, T1, T2, T3 ;

        /* convert a blockSizeX x blockSizeY block in the input image */
        switch (pixelFormat) {
          case pixelFormatRGB :
          case pixelFormatRGBA :
          case pixelFormatBGR :
          case pixelFormatBGRA :
		  case pixelFormatBGRAasL :
            // load 4x4 RGB pixels
            p0 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;
            p1 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;
            p2 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;
            p3 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;

            // transpose pixels as 32-bit integers (see also below)
            T0 = _mm_unpacklo_epi32(p0, p1);
            T1 = _mm_unpacklo_epi32(p2, p3);
            T2 = _mm_unpackhi_epi32(p0, p1);
            T3 = _mm_unpackhi_epi32(p2, p3);
            p0 = _mm_unpacklo_epi64(T0, T1);
            p1 = _mm_unpackhi_epi64(T0, T1);
            p2 = _mm_unpacklo_epi64(T2, T3);
            p3 = _mm_unpackhi_epi64(T2, T3);

            // store r
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p0, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p1, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p2, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p3, mask))) ;

			if (pixelFormat == pixelFormatBGRAasL) break ;

            // store g
            r += (imageWidth - 3) * imageHeight ;
            p0 = _mm_srli_epi32 (p0, 8) ;
            p1 = _mm_srli_epi32 (p1, 8) ;
            p2 = _mm_srli_epi32 (p2, 8) ;
            p3 = _mm_srli_epi32 (p3, 8) ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p0, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p1, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p2, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p3, mask))) ;

            // store b
            r += (imageWidth - 3) * imageHeight ;
            p0 = _mm_srli_epi32 (p0, 8) ;
            p1 = _mm_srli_epi32 (p1, 8) ;
            p2 = _mm_srli_epi32 (p2, 8) ;
            p3 = _mm_srli_epi32 (p3, 8) ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p0, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p1, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p2, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p3, mask))) ;
            break ;

          case pixelFormatL:
            // load 4x16 L pixels
            p0 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;
            p1 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;
            p2 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;
            p3 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;

            /*
             Pixels are collected in little-endian order: the first pixel
             is at the `right' (least significant byte of p0:

             p[0] = a, p[1] = b, ...

             p0: [ ... | ... | ... | d c b a ]
             p1: [ ... | ... | ... | h g f e ]
             p2: [ ... | ... | ... | l k j i ]
             p3: [ ... | ... | ... | p o n m ]

             The goal is to transpose four 4x4 subblocks in the
             4 x 16 pixel array. The first step interlaves individual
             pixels in p0 and p1:

             T0: [ ... | ... | h d g c | f b e a ]
             T1: [ ... | ... | p l o k | n j m i ]
             T2: [ ... | ... | ... | ... ]
             T3: [ ... | ... | ... | ... ]

             The second step interleaves groups of two pixels:

             p0: [pl hd | ok gc | nj fb | mi ea] (pixels in the rightmost 4x4 subblock)
             p1: ...
             p2: ...
             p3: ...

             The third step interlevaes groups of four pixels:

             T0: [ ... | njfb | ... | miea ]
             T1: ...
             T2: ...
             T3: ...

             The last step interleaves groups of eight pixels:

             p0: [ ... | ... | ... | miea ]
             p1: [ ... | ... | ... | njfb ]
             p2: [ ... | ... | ... | okgc ]
             p3: [ ... | ... | ... | dklp ]

             */

            T0 = _mm_unpacklo_epi8(p0, p1);
            T1 = _mm_unpacklo_epi8(p2, p3);
            T2 = _mm_unpackhi_epi8(p0, p1);
            T3 = _mm_unpackhi_epi8(p2, p3);
            p0 = _mm_unpacklo_epi16(T0, T1);
            p1 = _mm_unpackhi_epi16(T0, T1);
            p2 = _mm_unpacklo_epi16(T2, T3);
            p3 = _mm_unpackhi_epi16(T2, T3);
            T0 = _mm_unpacklo_epi32(p0, p1);
            T1 = _mm_unpacklo_epi32(p2, p3);
            T2 = _mm_unpackhi_epi32(p0, p1);
            T3 = _mm_unpackhi_epi32(p2, p3);
            p0 = _mm_unpacklo_epi64(T0, T1);
            p1 = _mm_unpackhi_epi64(T0, T1);
            p2 = _mm_unpacklo_epi64(T2, T3);
            p3 = _mm_unpackhi_epi64(T2, T3);

            // store four 4x4 subblock
            for (int i = 0 ; i < 4 ; ++i) {
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p0, shuffleL))) ; r += imageHeight ;
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p1, shuffleL))) ; r += imageHeight ;
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p2, shuffleL))) ; r += imageHeight ;
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p3, shuffleL))) ; r += imageHeight ;
              p0 = _mm_srli_si128 (p0, 4) ;
              p1 = _mm_srli_si128 (p1, 4) ;
              p2 = _mm_srli_si128 (p2, 4) ;
              p3 = _mm_srli_si128 (p3, 4) ;
            }
            break ;
        }
      } /* next y */

    boundary:
      /* special case if there is not a full 4x4 block to process */
      for ( ; y < imageHeight ; y += blockSizeY) {
        int bsy = (std::min)(imageHeight - y, blockSizeY) ;
        float * __restrict r ;
        float * rend ;
        for (int dx = 0 ; dx < bsx ; ++dx) {
          char unsigned const * __restrict pixel = rgb + y * rowStride + (x + dx) * pixelStride ;
          r = imageMemoryX + y + dx * imageHeight ;
          rend = r + bsy ;
          while (r != rend) {
            switch (pixelFormat) {
              case pixelFormatRGBA:
              case pixelFormatRGB:
                r[0 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[2 * imagePlaneStride] = (float) pixel[2] ;
                break ;
              case pixelFormatBGR:
              case pixelFormatBGRA:
                r[2 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[0 * imagePlaneStride] = (float) pixel[2] ;
                break;
			  case pixelFormatBGRAasL:
              case pixelFormatL:
                r[0] = (float) pixel[0] ;
                break ;
            }
            r += 1 ;
            pixel += rowStride ;
          }
        }
      }
    }
  }


#endif

} }
