// @file imread.hpp
// @brief Image reader
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__imread__
#define __vl__imread__

namespace vl {

#define VL_IMAGE_ERROR_MSG_MAX_LENGTH 256

  struct Image
  {
    int width ;
    int height ;
    int depth ;
    float * memory ;
    int error ;
    char errorMessage [VL_IMAGE_ERROR_MSG_MAX_LENGTH] ;

    Image() : width(0), height(0), depth(0), memory(0), error(0) { errorMessage[0] = 0 ; }
  } ;

  class ImageReader
  {
  public:
    ImageReader() ;
    ~ImageReader() ;
    Image read(char const * fileName, float * memory = 0) ;
    Image readDimensions(char const * fileName) ;

  private:
    class Impl ;
    Impl * impl ;
  } ;
}

#endif
