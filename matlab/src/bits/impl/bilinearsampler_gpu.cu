// defines the CUDA kernels and the dispatcher for them:
#include "bilinearsampler.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>
#include <cstdio>

// maximum size of each grid dimension:
#define MAX_GRID_DIM 65535 // this is probably a bad idea..

// an implementation of atomicAdd() for double (really slow)
static __device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

/* 2D grid of 1D blocks. */
__device__ int getGlobalIdx_2D_1D()
{
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId ;
}

template<typename type, bool backwardData, bool backwardGrid>
__global__ void forward_backward_kernel
(type* output,
 type* derData,
 type* derGrid,
 type const* data,
 type const* grid,
 type const* derOutput,
 int outHeight, int outWidth, int outDepth, int outCardinality,
 int inHeight, int inWidth, int inCardinality)
{
  const int offset = getGlobalIdx_2D_1D();
  const int nOut = outWidth * outHeight * outDepth * outCardinality ;
  if (offset >= nOut) { return ; }
  bool backward = backwardData | backwardGrid ;

  // get the index of the output image, feature channel, and pixel
  int k = offset ;
  int c = k / (outHeight * outWidth) ;
  int n = c / outDepth ;
  k %= (outHeight * outWidth) ;
  c %= outDepth ;

  // get the index of the input image
  int groupSize = outCardinality / inCardinality ;
  int nInputImage = n / groupSize ;
  int inputOffset = (inHeight * inWidth)*(outDepth * nInputImage + c) ;
  int gridOffset = 2 * ((outHeight * outWidth) * n + k) ; //+ 1;
  //int gridOffset = 2*k+1 ;

  // get the grid for this output image
  type py = grid[gridOffset + 0] ;
  type px = grid[gridOffset + 1] ;

  py = type(0.5)*(py + type(1.0)) * (inHeight - 1) ;
  px = type(0.5)*(px + type(1.0)) * (inWidth - 1) ;
  const int sx = floor(px); // todo: check floor vs floorf
  const int sy = floor(py);

  type acc = 0 ;
  type dgridx = 0 ;
  type dgridy = 0 ;
  type dy ;
  if (!backward) {
    data += inputOffset ;
  }
  if (backwardData) {
    derData += inputOffset ;
  }
  if (backward) {
    dy = derOutput[offset] ;
  }

  // todo: check boundary conditions in other frameworks and make
  // them the same
  if (0 <= sy && sy < inHeight && 0 <= sx && sx < inWidth) {
    // get the interpolation weights
    const type wx = px - sx ;
    const type wy = py - sy ;

    #pragma unroll
    for (int j=0; j < 2; j++) {
      #pragma unroll
      for (int i=0; i < 2; i++) {
        int ssy = sy + i ;
        int ssx = sx + j ;
        if (ssy < 0 || ssy >= inHeight || ssx < 0 || ssx >= inWidth) {
          continue ;
        }
        type wwx = (1-j)*(1-wx) + j*wx ;
        type wwy = (1-i)*(1-wy) + i*wy ;
        type ww = wwx * wwy ;
        if (!backward) {
          acc += ww * data[ssy + ssx * inHeight];
        } else {
          if (backwardData) {
            atomicAdd(derData  + ssy + ssx * inHeight, ww * dy) ;
          }
          if (backwardGrid) {
            type x = data[ssy + ssx * inHeight] ;
            dgridy += wwy * dy * x ;
            dgridx += wwx * dy * x ;
          }
        }
      }
    }
    if (!backward) {
      output[offset] = acc ;
    }
    if (backwardGrid) {
      derGrid[gridOffset + 0] = dgridy ;
      derGrid[gridOffset + 1] = dgridx ;
    }
  }
}

/** get the number of threads (1D) and blocks (2D). **/
vl::Error get_launch_params(const int& N, int& nTh, int& nGx, int& nGy)
{
  nGx = vl::divideUpwards(N, VL_CUDA_NUM_THREADS);
  if (nGx == 1) {
    nTh = N;
    nGy = 1;
  } else {
    nTh = VL_CUDA_NUM_THREADS;
    if (nGx <= MAX_GRID_DIM) {
      nGy = 1;
    } else {
      nGy = vl::divideUpwards(nGx, MAX_GRID_DIM);
      nGx = MAX_GRID_DIM;
      if (nGy > MAX_GRID_DIM) {
        // the following print statement is probably not
        // shown in the matlab JVM console:
        std::printf("BilinearSamper: output volume should be smaller.");
        return vl::vlErrorCuda;
      }
    }
  }
  return vl::vlSuccess;
}

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

  if (backwardData) {
    //memset(derData, 0, inHeight * inWidth * outDepth * inCardinality * sizeof(type)) ;
  }

  // setup and launch the kernel for DER-DATA:
  const int outVolume = outHeight * outWidth * outDepth * outCardinality ;
  int nTh, nGx, nGy;
  vl::Error volume_ok = get_launch_params(outVolume, nTh, nGx, nGy);
  if (volume_ok != vl::vlSuccess) { return volume_ok;}

  dim3  gridDim(nGx,nGy); // grid-dimensions
  forward_backward_kernel <type, backwardData, backwardGrid>
    <<< gridDim, nTh >>> (output,
                          derData,
                          derGrid,
                          data,
                          grid,
                          derOutput,
                          outHeight, outWidth, outDepth, outCardinality,
                          inHeight, inWidth, inCardinality) ;

  cudaError_t status = cudaPeekAtLastError() ;
  return (status != cudaSuccess) ? vl::vlErrorCuda : vl::vlSuccess ;
}

namespace vl { namespace impl {

  template<typename type>
  struct bilinearsampler<vl::GPU, type>
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
      } else if (derData == NULL && derGrid) {
        DISPATCH(false, true) ;
      } else if (derData && derGrid) {
        DISPATCH(true, true) ;
      }
      return error ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::bilinearsampler<vl::GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bilinearsampler<vl::GPU, double> ;
#endif

