/** @file vl_nnconv.cu
 ** @brief Convolution block
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnhelper.h"
#include "bits/im2col.hpp"

#include <assert.h>

#include <blas.h>
#ifdef ENABLE_GPU
#include <cublas_v2.h>
#endif

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_verbose,
  opt_no_der_data,
  opt_no_der_filters,
  opt_no_der_biases,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride             },
  {"Pad",              1,   opt_pad                },
  {"Verbose",          0,   opt_verbose            },
  {"NoDerData",        0,   opt_no_der_data        },
  {"NoDerFilters",     0,   opt_no_der_filters     },
  {"NoDerBiases",      0,   opt_no_der_biases      },
  {0,                  0,   0                      }
} ;

/* ---------------------------------------------------------------- */
/*                                                  Dispatcher func */
/* ---------------------------------------------------------------- */

#ifdef ENABLE_GPU
cublasHandle_t thisCublasHandle ;
#endif

static void
sgemv_dispatch(bool gpuMode,
               char op,
               ptrdiff_t m, ptrdiff_t n,
               float alpha,
               float const * a, ptrdiff_t lda,
               float const * x, ptrdiff_t incx,
               float beta,
               float * y, ptrdiff_t incy)
{
  if (!gpuMode) {
    sgemv(&op,
          &m, &n, &alpha,
          (float*)a, &lda,
          (float*)x, &incx,
          &beta,
          y, &incy) ;
  } else {
#ifdef ENABLE_GPU
    cublasSgemv(thisCublasHandle,
                (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                (int)m, (int)n,
                &alpha,
                a, lda,
                x, (int)incx,
                &beta,
                y, (int)incy) ;
#endif
  }
}

static void
sgemm_dispatch(bool gpuMode,
               char op1, char op2,
               ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
               float alpha,
               float const * a, ptrdiff_t lda,
               float const * b, ptrdiff_t ldb,
               float beta,
               float * c, ptrdiff_t ldc)
{
  if (!gpuMode) {
    sgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (float*)a, &lda,
          (float*)b, &ldb,
          &beta,
          c, &ldc) ;
  } else {
#ifdef ENABLE_GPU
    cublasSgemm(thisCublasHandle,
                (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                (int)m, (int)n, (int)k,
                &alpha,
                a, (int)lda,
                b, (int)ldb,
                &beta,
                c, (int)ldc);
#endif
  }
}

/* ---------------------------------------------------------------- */
/*                                                            Cache */
/* ---------------------------------------------------------------- */

bool persistentDataInitialized = false ;
PackedData temp ;
PackedData allOnes ;

void atExit()
{
  packed_data_deinit (&temp)  ;
  packed_data_deinit (&allOnes)  ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_FILTERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERFILTERS, OUT_DERBIASES, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  /* inputs */
  PackedData data ;
  PackedData filters ;
  PackedData biases ;
  PackedData derOutput ;

  /* outputs */
  PackedData output ;
  PackedData derData  ;
  PackedData derFilters ;
  PackedData derBiases ;

  PackedDataGeometry outputGeom ;
  PackedDataGeometry derDataGeom  ;
  PackedDataGeometry derFiltersGeom ;
  PackedDataGeometry derBiasesGeom ;
  PackedDataGeometry tempGeom ;
  PackedDataGeometry allOnesGeom ;

  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  int numGroups = 1 ;

#if ENABLE_GPU
  cublasStatus_t stat;
  bool gpuMode = false ;
#else
  bool const gpuMode = false ;
#endif
  bool backMode = false ;
  bool biasMode = false ;
  bool fullyConnectedMode = false ;
  bool computeOutput = true ;
  bool computeDerData = true ;
  bool computeDerFilters = true ;
  bool computeDerBiases = true ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  packed_data_init_empty(&data) ;
  packed_data_init_empty(&filters) ;
  packed_data_init_empty(&biases) ;
  packed_data_init_empty(&derOutput) ;
  packed_data_init_empty(&output) ;
  packed_data_init_empty(&derData) ;
  packed_data_init_empty(&derFilters) ;
  packed_data_init_empty(&derBiases) ;
  if (!persistentDataInitialized) {
    persistentDataInitialized = true ;
    packed_data_init_empty(&temp) ;
    packed_data_init_empty(&allOnes) ;
  }

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 3) {
    mexErrMsgTxt("There are less than three arguments.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  biasMode = (mxGetNumberOfElements(in[IN_BIASES]) > 0) ;

#if ENABLE_GPU
  gpuMode = mxIsGPUArray(in[IN_DATA]) ;
  if (gpuMode) {
    mxInitGPU() ;
    stat = cublasCreate(&thisCublasHandle) ;
    if (stat != CUBLAS_STATUS_SUCCESS) {
      mexErrMsgTxt("Could not initialize cuBLAS.") ;
    }
  }
#else
  if (!mxIsNumeric(in[IN_DATA])) {
    mexErrMsgTxt("DATA must be numeric (note: GPU support not compiled).") ;
  }
#endif

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            break ;
          case 2:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            break ;
          case 4:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_no_der_data :
        computeDerData = VL_FALSE ;
        break ;

      case opt_no_der_filters :
        computeDerFilters = VL_FALSE ;
        break ;

      case opt_no_der_biases :
        computeDerBiases = VL_FALSE ;
        break ;
        
      default: break ;
    }
  }

  packed_data_init_with_array (&data, gpuMode, in[IN_DATA]) ;
  packed_data_init_with_array (&filters, gpuMode, in[IN_FILTERS]) ;
  if (biasMode) { packed_data_init_with_array(&biases, gpuMode, in[IN_BIASES]) ; }
  if (backMode) { packed_data_init_with_array(&derOutput, gpuMode, in[IN_DEROUTPUT]) ; }

  if (data.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (filters.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("FILTERS is not of class SINGLE.");
  }
  if (biasMode && (biases.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("BIASES is not of class SINGLE.");
  }
  if (backMode && (derOutput.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("DEROUTPUT is not of class SINGLE.");
  }

  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }

  packed_data_geom_init(&outputGeom,
                        mxSINGLE_CLASS,
                        (data.geom.height + (padTop+padBottom) - filters.geom.height)/strideY + 1,
                        (data.geom.width + (padLeft+padRight) - filters.geom.width)/strideX + 1,
                        filters.geom.size,
                        data.geom.size) ;

  /* grouped filters */
  numGroups = data.geom.depth / filters.geom.depth ;

  /* if the output is 1x1 pixels, then there is no need to actually
   call im2col as it does not do anything
   */
  fullyConnectedMode = (outputGeom.height == 1 &&
                        outputGeom.width == 1 &&
                        padTop == 0 &&
                        padBottom == 0 &&
                        padLeft == 0 &&
                        padRight == 0 &&
                        numGroups == 1) ;

  derDataGeom = data.geom ;
  derFiltersGeom = filters.geom ;
  if (biasMode) {
    if (fullyConnectedMode) {
      packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                             1, 1,
                             1, data.geom.size) ;
    } else {
      packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                             outputGeom.height,
                             outputGeom.width,
                             1, 1) ;
    }
    derBiasesGeom = biases.geom ;
  } else {
    packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                           0, 0, 0, 0) ;
  }

  packed_data_geom_init
  (&tempGeom,
   mxSINGLE_CLASS,
   outputGeom.height,
   outputGeom.width,
   filters.geom.height*filters.geom.width*filters.geom.depth*numGroups,
   1) ;

  if (verbosity > 0) {
    mexPrintf("vl_nnconv: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnconv: stride: [%d %d], pad: [%d %d %d %d], numGroups: %d, bias: %d, fully connected: %d\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight,
              numGroups, biasMode, fullyConnectedMode) ;
    packed_data_geom_display(&data.geom, "vl_nnconv: data") ;
    packed_data_geom_display(&filters.geom, "vl_nnconv: filters") ;
    if (biasMode) { packed_data_geom_display(&biases.geom, "vl_nnconv: biases") ; }
    if (backMode) {
      packed_data_geom_display(&derOutput.geom, "vl_nnconv: derOutput") ;
      packed_data_geom_display(&derDataGeom, "vl_nnconv: derData") ;
      packed_data_geom_display(&derFiltersGeom, "vl_nnconv: derFilters") ;
      if (biasMode) { packed_data_geom_display(&derBiasesGeom, "vl_nnconv: derBiases") ; }
    } else {
      packed_data_geom_display(&outputGeom, "vl_nnconv: output") ;
    }
    packed_data_geom_display(&tempGeom, "vl_nnconv: temp") ;
    packed_data_geom_display(&temp.geom, "vl_nnconv: temp (cached)") ;
    packed_data_geom_display(&allOnesGeom, "vl_nnconv: allOnes") ;
    packed_data_geom_display(&allOnes.geom, "vl_nnconv: allOnes (cached)") ;
  }

  if (backMode) {
    if (derOutput.geom.height != tempGeom.height ||
        derOutput.geom.width != tempGeom.width ||
        derOutput.geom.depth != filters.geom.size ||
        derOutput.geom.size != data.geom.size)
    {
      mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and FILTERS.") ;
    }
  }

  if (numGroups * filters.geom.depth != data.geom.depth) {
    mexErrMsgTxt("The filter depth does not divide the image depth.") ;
  }

  if (filters.geom.size % numGroups != 0) {
    mexErrMsgTxt("The number of filter groups does not divide the total number of filters.") ;
  }

  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }

  if (data.geom.height + (padTop+padBottom) < filters.geom.height ||
      data.geom.width + (padLeft+padRight) < filters.geom.width) {
    mexErrMsgTxt("FILTERS are larger than the DATA (including padding).") ;
  }

  if (filters.geom.height == 0 || filters.geom.width == 0 || filters.geom.depth == 0) {
    mexErrMsgTxt("A dimension of FILTERS is void.") ;
  }

  if (biasMode) {
    if (biases.geom.numElements != filters.geom.size) {
      mexErrMsgTxt("The number of elements of BIASES is not the same as the number of filters.") ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  /* auxiliary buffers */
  if (biasMode) {
    if (allOnes.memorySize < allOnesGeom.numElements * sizeof(float) ||
        (allOnes.mode == matlabGpuArray || allOnes.mode == cudaMallocMemory) != gpuMode) {
      packed_data_deinit (&allOnes) ;
      packed_data_init_with_geom (&allOnes, gpuMode, allOnesGeom, true, true, 1.0f) ;
    }
  }
  if (!fullyConnectedMode) {
    if (temp.memorySize < tempGeom.numElements * sizeof(float) ||
        (temp.mode == matlabGpuArray || temp.mode == cudaMallocMemory) != gpuMode) {
      packed_data_deinit (&temp) ;
      packed_data_init_with_geom (&temp, gpuMode, tempGeom, true, false, 0);
    }
  }
  if (!backMode && computeOutput) {
    packed_data_init_with_geom(&output, gpuMode, outputGeom, false, false, 0) ;
  }
  if (backMode && computeDerData) {
    packed_data_init_with_geom(&derData, gpuMode, derDataGeom, false, false, 0) ;
  }
  if (backMode && computeDerFilters) {
    packed_data_init_with_geom(&derFilters, gpuMode, derFiltersGeom, false, false, 0) ;
  }
  if (backMode && computeDerBiases && biasMode) {
    packed_data_init_with_geom(&derBiases, gpuMode, derBiasesGeom, false, false, 0) ;
  }

  if (fullyConnectedMode) {
    float alpha = 1 ;
    float beta = 0 ;
    ptrdiff_t filtersVolume = filters.geom.height*filters.geom.width*filters.geom.depth ;

    /* especially optimized */
    if (!backMode) {
      if (data.geom.size == 1) {
        /* one image in the stack */
        sgemv_dispatch(gpuMode, 't',
                       filtersVolume, filters.geom.size,
                       alpha,
                       filters.memory, filtersVolume,
                       data.memory, 1,
                       beta,
                       output.memory, 1) ;
      } else {
        /* multiple images in the stack */
        sgemm_dispatch(gpuMode, 't', 'n',
                       filters.geom.size, data.geom.size, filtersVolume,
                       alpha,
                       filters.memory, filtersVolume,
                       data.memory, filtersVolume,
                       beta,
                       output.memory, filters.geom.size) ;
      }
      if (biasMode) {
        float beta = 1 ;
        ptrdiff_t q = 1 ;
        sgemm_dispatch(gpuMode, 'n', 'n',
                        filters.geom.size, data.geom.size, q,
                        alpha,
                        biases.memory, filters.geom.size,
                        allOnes.memory, q,
                        beta,
                        output.memory, filters.geom.size) ;
      }
    } else {
      /* back mode */
      if (computeDerFilters) {
        sgemm_dispatch(gpuMode, 'n', 't',
                       filtersVolume, filters.geom.size, data.geom.size,
                       alpha,
                       data.memory, filtersVolume,
                       derOutput.memory, filters.geom.size,
                       beta,
                       derFilters.memory, filtersVolume) ;
      }
      if (computeDerBiases & biasMode) {
        ptrdiff_t q = 1 ;
        sgemm_dispatch(gpuMode, 'n', 't',
                       q, filters.geom.size, data.geom.size,
                       alpha,
                       allOnes.memory, q,
                       derOutput.memory, filters.geom.size,
                       beta,
                       derBiases.memory, q) ;
      }
      if (computeDerData) {
        sgemm_dispatch(gpuMode, 'n', 'n',
                       filtersVolume, data.geom.size, filters.geom.size,
                       alpha,
                       filters.memory, filtersVolume,
                       derOutput.memory, filters.geom.size,
                       beta,
                       derData.memory, filtersVolume) ;
      }
    }
  } else {
    /* not fully connected */
    for (int image = 0 ; image < data.geom.size ; ++image) {
      /*
        temp (phi(x)): m x k
        filters, derFilters: k x n (for one group of filters)
        derOutput (dzdy) : m x n (for one group of filters)
        res (y) : m x n (for one group of filters)
      */
      ptrdiff_t dataOffset = (data.geom.height*data.geom.width*data.geom.depth) * image ;
      ptrdiff_t outputOffset = (output.geom.height*output.geom.width*output.geom.depth) * image ;
      ptrdiff_t derDataOffset = (derData.geom.height*derData.geom.width*derData.geom.depth) * image ;
      ptrdiff_t derOutputOffset = (derOutput.geom.height*derOutput.geom.width*derOutput.geom.depth) * image ;
      ptrdiff_t m = tempGeom.height * tempGeom.width ; /* num output pixels */
      ptrdiff_t n = filters.geom.size/numGroups ; /* num filters per group */
      ptrdiff_t k = filters.geom.height*filters.geom.width*filters.geom.depth ; /* filter volume */

      if (backMode) {
        /* ---------------------------------------------------------- */
        /*                                              Backward mode */
        /* ---------------------------------------------------------- */

        /* compute derFilters dz/dF */
        if (computeDerFilters) {
          if (!fullyConnectedMode) {
            if (gpuMode) {
#ifdef ENABLE_GPU
              im2col_gpu<float>(temp.memory,
                                data.memory + dataOffset,
                                data.geom.height, data.geom.width, data.geom.depth,
                                filters.geom.height, filters.geom.width,
                                strideY, strideX,
                                padTop, padBottom, padLeft, padRight) ;
#else
              assert(false) ;
#endif
            } else {
              im2col_cpu<float>(temp.memory,
                                data.memory + dataOffset,
                                data.geom.height, data.geom.width, data.geom.depth,
                                filters.geom.height, filters.geom.width,
                                strideY, strideX,
                                padTop, padBottom, padLeft, padRight) ;
            }
          }
          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = m * k * g ;
            ptrdiff_t derOutputGrpOffset = m * n * g ;
            float alpha = 1 ;
            float beta = (image > 0)  ;
            sgemm_dispatch(gpuMode, 't', 'n',
                           k, n, m,
                           alpha,
                           (fullyConnectedMode ? data.memory : temp.memory)
                           + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, m,
                           derOutput.memory + derOutputOffset + derOutputGrpOffset, m,
                           beta,
                           derFilters.memory + filterGrpOffset, k) ;
          }
        }

        /* compute derData dz/dbias */
        if (computeDerBiases & biasMode) {
          sgemv_dispatch(gpuMode, 't',
                         m, filters.geom.size,
                         1, /* alpha */
                         derOutput.memory + derOutputOffset, m,
                         allOnes.memory, 1,
                         (float)(image > 0), /* beta */
                         derBiases.memory, 1) ;
        }

        /* compute derData dz/dx */
        if (computeDerData) {
          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = m * k * g ;
            ptrdiff_t derOutputGrpOffset = m * n * g ;
            float alpha = 1 ;
            float beta = fullyConnectedMode ? (g > 0) : 0 ;
            sgemm_dispatch(gpuMode, 'n', 't',
                           m, k, n,
                           alpha,
                           derOutput.memory + derOutputOffset + derOutputGrpOffset, m,
                           filters.memory + filterGrpOffset, k,
                           beta,
                           (fullyConnectedMode ? derData.memory : temp.memory)
                           + (fullyConnectedMode ? + derDataOffset : 0) + tempGrpOffset,
                           m) ;
          }
#if 1
          if (!fullyConnectedMode) {
            if (gpuMode) {
#ifdef ENABLE_GPU
              col2im_gpu<float>(derData.memory + derDataOffset,
                                temp.memory,
                                data.geom.height, data.geom.width, data.geom.depth,
                                filters.geom.height, filters.geom.width,
                                strideY, strideX,
                                padTop, padBottom, padLeft, padRight) ;
#else
              assert(false) ;
#endif
            } else {
              col2im_cpu<float>(derData.memory + derDataOffset,
                                temp.memory,
                                data.geom.height, data.geom.width, data.geom.depth,
                                filters.geom.height, filters.geom.width,
                                strideY, strideX,
                                padTop, padBottom, padLeft, padRight) ;
            }
          }
#endif
        }
      } else {
        /* ---------------------------------------------------------- */
        /*                                               Forward mode */
        /* ---------------------------------------------------------- */
        if (computeOutput) {
          if (gpuMode) {
#ifdef ENABLE_GPU
            im2col_gpu<float>(temp.memory,
                              data.memory + dataOffset,
                              data.geom.height, data.geom.width, data.geom.depth,
                              filters.geom.height, filters.geom.width,
                              strideY, strideX,
                              padTop, padBottom, padLeft, padRight) ;
#else
            assert(false) ;
#endif
          } else {
            im2col_cpu<float>(temp.memory,
                              data.memory + dataOffset,
                              data.geom.height, data.geom.width, data.geom.depth,
                              filters.geom.height, filters.geom.width,
                              strideY, strideX,
                              padTop, padBottom, padLeft, padRight) ;
          }
        }
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = k * n * g ;
          ptrdiff_t tempGrpOffset = m * k * g ;
          ptrdiff_t outputGrpOffset = m * n * g  ;
          float alpha = 1 ;
          float beta = 0 ;
          sgemm_dispatch(gpuMode, 'n', 'n',
                         m, n, k,
                         alpha,
                         (fullyConnectedMode ? data.memory : temp.memory)
                         + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, m,
                         filters.memory + filterGrpOffset, k,
                         beta,
                         output.memory + outputOffset + outputGrpOffset, m) ;
        }
        if (biasMode) {
          float alpha = 1 ;
          float beta = 1 ;
          ptrdiff_t q = 1 ;
          sgemm_dispatch(gpuMode, 'n', 'n',
                         m, biases.geom.numElements, q,
                         alpha,
                         allOnes.memory, m,
                         biases.memory, q,
                         beta,
                         output.memory + outputOffset, m) ;
        }
      }
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

#ifdef ENABLE_GPU
  if (gpuMode) { cublasDestroy(thisCublasHandle) ; }
#endif

  packed_data_deinit(&data) ;
  packed_data_deinit(&filters) ;
  if (biasMode) { packed_data_deinit(&biases) ; }
  if (backMode) {
    packed_data_deinit(&derOutput) ;
    out[OUT_RESULT] = (computeDerData) ? packed_data_deinit_extracting_array(&derData) : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERFILTERS] =(computeDerFilters)? packed_data_deinit_extracting_array(&derFilters) : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERBIASES] = (computeDerBiases & biasMode) ? packed_data_deinit_extracting_array(&derBiases) : mxCreateDoubleMatrix(0,0,mxREAL) ;
  } else {
    out[OUT_RESULT] = packed_data_deinit_extracting_array(&output) ;
  }
}
