// @file imread_quartz.cpp
// @brief Image reader based on Apple Quartz (Core Graphics).
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../imread.hpp"
#include "imread_helpers.hpp"

#import <ImageIO/ImageIO.h>
#include <algorithm>

#include <iostream>

/* ---------------------------------------------------------------- */
/*                                     Quartz reader implementation */
/* ---------------------------------------------------------------- */

#define check(x) \
if (!(x)) { image.error = 1 ; goto done ; }

vl::ImageReader::ImageReader()
: impl(NULL)
{ }

vl::ImageReader::~ImageReader()
{ }

vl::Image
vl::ImageReader::read(const char * fileName, float * memory)
{
  // intermediate buffer
  char unsigned * pixels = NULL ;
  int bytesPerPixel ;
  int bytesPerRow ;

  // Core graphics
  CGBitmapInfo bitmapInfo ;
  CFURLRef url = NULL ;
  CGImageSourceRef imageSourceRef = NULL ;
  CGImageRef imageRef = NULL ;
  CGContextRef contextRef = NULL ;
  CGColorSpaceRef sourceColorSpaceRef = NULL ;
  CGColorSpaceRef colorSpaceRef = NULL ;

  // initialize the image as null
  Image image ;
  image.width = 0 ;
  image.height = 0 ;
  image.depth = 0 ;
  image.memory = NULL ;
  image.error = 0 ;

  // get file
  url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, (const UInt8 *)fileName, strlen(fileName), false) ;
  check(url) ;

  // get image source from file
  imageSourceRef = CGImageSourceCreateWithURL(url, NULL) ;
  check(imageSourceRef) ;

  // get image from image source
  imageRef = CGImageSourceCreateImageAtIndex(imageSourceRef, 0, NULL);
  check(imageRef) ;

  sourceColorSpaceRef = CGImageGetColorSpace(imageRef) ;
  check(sourceColorSpaceRef) ;

  image.width = CGImageGetWidth(imageRef);
  image.height = CGImageGetHeight(imageRef);
  image.depth = CGColorSpaceGetNumberOfComponents(sourceColorSpaceRef) ;
  check(image.depth == 1 || image.depth == 3) ;

  // decode image to L (8 bits per pixel) or RGBA (32 bits per pixel)
  switch (image.depth) {
    case 1:
      colorSpaceRef = CGColorSpaceCreateDeviceGray();
      bytesPerPixel = 1 ;
      bitmapInfo = kCGImageAlphaNone ;
      break ;

    case 3:
      colorSpaceRef = CGColorSpaceCreateDeviceRGB();
      bytesPerPixel = 4 ;
      bitmapInfo = kCGImageAlphaPremultipliedLast || kCGBitmapByteOrder32Big ;
      /* this means
       pixels[0] = R
       pixels[1] = G
       pixels[2] = B
       pixels[3] = A
       */
      break ;

  }
  check(colorSpaceRef) ;

  bytesPerRow = image.width * bytesPerPixel ;
  pixels = (char unsigned*)malloc(image.height * bytesPerRow) ;
  check(pixels) ;

  contextRef = CGBitmapContextCreate(pixels,
                                     image.width, image.height,
                                     8, bytesPerRow,
                                     colorSpaceRef,
                                     bitmapInfo) ;
  check(contextRef) ;

  CGContextDrawImage(contextRef, CGRectMake(0, 0, image.width, image.height), imageRef);

  // copy pixels to MATLAB format
  if (memory == NULL) {
    image.memory = (float*)malloc(image.height * image.width * image.depth * sizeof(float)) ;
    check(image.memory) ;
  } else {
    image.memory = memory ;
  }
  switch (image.depth) {
    case 3:
      vl::impl::imageFromPixels<impl::pixelFormatRGBA>(image, pixels, image.width * bytesPerPixel) ;
      break ;
    case 1:
      vl::impl::imageFromPixels<impl::pixelFormatL>(image, pixels, image.width * bytesPerPixel) ;
      break ;
  }

done:
  if (pixels) { free(pixels) ; }
  if (contextRef) { CFRelease(contextRef) ; }
  if (colorSpaceRef) { CFRelease(colorSpaceRef) ; }
  if (imageRef) { CFRelease(imageRef) ; }
  if (imageSourceRef) { CFRelease(imageSourceRef) ; }
  if (url) { CFRelease(url) ; }
  return image ;
}

vl::Image
vl::ImageReader::readDimensions(const char * fileName)
{
  // intermediate buffer
  char unsigned * rgba = NULL ;

  // Core graphics
  CFURLRef url = NULL ;
  CGImageSourceRef imageSourceRef = NULL ;
  CGImageRef imageRef = NULL ;
  CGColorSpaceRef colorSpaceRef = NULL ;

  // initialize the image as null
  Image image ;
  image.width = 0 ;
  image.height = 0 ;
  image.depth = 0 ;
  image.memory = NULL ;
  image.error = 0 ;

  // get file
  url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, (const UInt8 *)fileName, strlen(fileName), false) ;
  check(url) ;

  // get image source from file
  imageSourceRef = CGImageSourceCreateWithURL(url, NULL) ;
  check(imageSourceRef) ;

  // get image from image source
  imageRef = CGImageSourceCreateImageAtIndex(imageSourceRef, 0, NULL);
  check(imageRef) ;

  colorSpaceRef = CGColorSpaceCreateDeviceRGB();
  check(colorSpaceRef) ;

  image.width = CGImageGetWidth(imageRef);
  image.height = CGImageGetHeight(imageRef);
  image.depth = CGColorSpaceGetNumberOfComponents(colorSpaceRef) ;
  check(image.depth == 1 || image.depth == 3) ;

done:
  if (colorSpaceRef) { CFRelease(colorSpaceRef) ; }
  if (imageRef) { CFRelease(imageRef) ; }
  if (imageSourceRef) { CFRelease(imageSourceRef) ; }
  if (url) { CFRelease(url) ; }
  return image ;
}
