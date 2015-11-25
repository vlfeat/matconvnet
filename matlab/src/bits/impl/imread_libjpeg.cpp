// @file imread_libjpeg.cpp
// @brief Image reader based on libjpeg.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../imread.hpp"
#include "imread_helpers.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
extern "C" {
#include <jpeglib.h>
#include <setjmp.h>
}

/* ---------------------------------------------------------------- */
/*                                    LibJPEG reader implementation */
/* ---------------------------------------------------------------- */

class vl::ImageReader::Impl
{
public:
  Impl() ;
  ~Impl() ;
  struct jpeg_error_mgr jpegErrorManager ; /* must be the first element */
  struct jpeg_decompress_struct decompressor ;
  jmp_buf onJpegError ;
  char jpegLastErrorMsg [JMSG_LENGTH_MAX] ;

  vl::Image read(char const * filename, float * memory) ;
  vl::Image readDimensions(char const * filename) ;


  static void reader_jpeg_error (j_common_ptr cinfo)
  {
    vl::ImageReader::Impl* self = (vl::ImageReader::Impl*) cinfo->err ;
    (*(cinfo->err->format_message)) (cinfo, self->jpegLastErrorMsg) ;
    longjmp(self->onJpegError, 1) ;
  }
} ;

vl::ImageReader::Impl::Impl()
{
  decompressor.err = jpeg_std_error(&jpegErrorManager) ;
  jpegErrorManager.error_exit = reader_jpeg_error ;
  jpeg_create_decompress(&decompressor) ;
  decompressor.out_color_space = JCS_RGB ;
  decompressor.quantize_colors = FALSE ;
}

vl::ImageReader::Impl::~Impl()
{
  jpeg_destroy_decompress(&decompressor) ;
}

vl::Image
vl::ImageReader::Impl::read(char const * filename, float * memory)
{
  int row_stride ;
  const int blockSize = 32 ;
  char unsigned * pixels = NULL ;
  JSAMPARRAY scanlines = NULL ;
  bool requiresAbort = false ;

  /* initialize the image as null */
  Image image ;
  image.width = 0 ;
  image.height = 0 ;
  image.depth = 0 ;
  image.memory = NULL ;
  image.error = 0 ;

  /* open file */
  FILE* fp = fopen(filename, "r") ;
  if (fp == NULL) {
    image.error = 1 ;
    return image ;
  }

  /* handle LibJPEG errors */
  if (setjmp(onJpegError)) {
    image.error = 1 ;
    goto done ;
  }

  /* set which file to read */
  jpeg_stdio_src(&decompressor, fp);

  /* read image metadata */
  jpeg_read_header(&decompressor, TRUE) ;
  requiresAbort = true ;

  /* get the output dimension (this may differ from the input if we were to scale the image) */
  jpeg_calc_output_dimensions(&decompressor) ;
  image.width = decompressor.output_width ;
  image.height = decompressor.output_height ;
  image.depth = decompressor.output_components ;

  /* allocate image memory */
  if (memory == NULL) {
    image.memory = (float*)malloc(sizeof(float)*image.depth*image.width*image.height) ;
  } else {
    image.memory = memory ;
  }
  if (image.memory == NULL) {
    image.error = 1 ;
    goto done ;
  }

  /* allocate scaline buffer */
  pixels = (char unsigned*)malloc(sizeof(char) * image.width * image.height * image.depth) ;
  if (pixels == NULL) {
    image.error = 1 ;
    goto done ;
  }
  scanlines = (char unsigned**)malloc(sizeof(char*) * image.height) ;
  if (scanlines == NULL) {
    image.error = 1 ;
    goto done ;
  }
  for (int y = 0 ; y < image.height ; ++y) { scanlines[y] = pixels + image.depth * image.width * y ; }

  /* decompress each scanline and transpose the result into MATLAB format */
  jpeg_start_decompress(&decompressor);
  while(decompressor.output_scanline < image.height) {
    jpeg_read_scanlines(&decompressor,
                        scanlines + decompressor.output_scanline,
                        image.height - decompressor.output_scanline);
  }
  switch (image.depth) {
    case 3 : vl::impl::imageFromPixels<impl::pixelFormatRGB>(image, pixels, image.width*3) ; break ;
    case 1 : vl::impl::imageFromPixels<impl::pixelFormatL>(image, pixels, image.width*1) ; break ;
  }
  jpeg_finish_decompress(&decompressor) ;
  requiresAbort = false ;

done:
  if (requiresAbort) { jpeg_abort((j_common_ptr)&decompressor) ; }
  if (scanlines) free(scanlines) ;
  if (pixels) free(pixels) ;
  fclose(fp) ;
  return image ;
}

vl::Image
vl::ImageReader::Impl::readDimensions(char const * filename)
{
  int row_stride ;
  const int blockSize = 32 ;
  char unsigned * pixels = NULL ;
  JSAMPARRAY scanlines ;
  bool requiresAbort = false ;

  // initialize the image as null
  Image image ;
  image.width = 0 ;
  image.height = 0 ;
  image.depth = 0 ;
  image.memory = NULL ;
  image.error = 0 ;

  // open file
  FILE* fp = fopen(filename, "r") ;
  if (fp == NULL) {
    image.error = 1 ;
    return image ;
  }

  // handle LibJPEG errors
  if (setjmp(onJpegError)) {
    image.error = 1 ;
    goto done ;
  }

  /* set which file to read */
  jpeg_stdio_src(&decompressor, fp);

  /* read image metadata */
  jpeg_read_header(&decompressor, TRUE) ;
  requiresAbort = true ;

  /* get the output dimension (this may differ from the input if we were to scale the image) */
  jpeg_calc_output_dimensions(&decompressor) ;
  image.width = decompressor.output_width ;
  image.height = decompressor.output_height ;
  image.depth = decompressor.output_components ;

done:
  if (requiresAbort) { jpeg_abort((j_common_ptr)&decompressor) ; }
  fclose(fp) ;
  return image ;
}

/* ---------------------------------------------------------------- */
/*                                                   LibJPEG reader */
/* ---------------------------------------------------------------- */

vl::ImageReader::ImageReader()
: impl(NULL)
{
  impl = new vl::ImageReader::Impl() ;
}

vl::ImageReader::~ImageReader()
{
  delete impl ;
}

vl::Image
vl::ImageReader::read(char const * filename, float * memory)
{
  return impl->read(filename, memory) ;
}

vl::Image
vl::ImageReader::readDimensions(char const * filename)
{
  return impl->readDimensions(filename) ;
}
