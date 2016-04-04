// defines the CUDA kernels and the dispatcher for them:
#include "bilinearsampler.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>
#include <cstdio>

// maximum size of each grid dimension:
#define MAX_GRID_DIM 65535 // this is probably a bad idea..

/* 2D grid of 1D blocks. */
__device__ int getGlobalIdx_2D_1D() {
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;        
  int threadId = blockId * blockDim.x + threadIdx.x; 
  return threadId;
}

// FORWARD KERNEL:
__global__ void bilinearsampler_fwd_kernel (float* const out,
                const float* const xin,
                const float* const grid,
                const int outHeight,
                const int outWidth,
                const int outSize,
                const int inHeight,
                const int inWidth,
                const int inDepth,
                const int inSize )  {

  // get the flat index of this output pixel:
  const int flatIndex = getGlobalIdx_2D_1D();
  const int nOut = outWidth * outHeight * inDepth * outSize;
  // thread boundary check:
  if (flatIndex >= nOut) { return; }

  // extract the indices of the output location 
  // that this thread is computing (ix,iy,ic,ib):  
  int ix = flatIndex; // x index
  int iy = ix / outHeight; // y index
  int ic = iy / outWidth; // channel index
  const int ib = ic / inDepth;  // image index (in the batch)
  ix %= outHeight; 
  iy %= outWidth;
  ic %= inDepth;

  // get the values of source-x and source-y from the
  // correct location in the grid:
  // grid has the size: outHeight x outWidth x 2 x nBatch
  const int igx = ix + iy*outHeight + ib * 2*outHeight*outWidth;
  const int igy = igx + outHeight*outWidth;
  // get the location from the grid:
  // and convert from [-1,1] --> [0,H-1]
  float px = grid[igx];
  float py = grid[igy];
  px = (px + 1.0) / 2.0 * (inHeight-1);
  py = (py + 1.0) / 2.0 * (inWidth-1);
  const int sx = floor(px);
  const int sy = floor(py);
  // pre-emptive check here for out of range:
  if (sx < -1 || sx > inHeight-1 || sy < -1 || sy > inWidth-1) {
    // this is out of range, write a zero:
    out[flatIndex] = 0;
    return;
  }
  // get the interpolation weights:
  const float wx = px - sx;
  const float wy = py - sy;
  // add the weighted sum to the output:
  int ssx, ssy;
  float outval = 0;

  // get the number of input-image from which we get the data:
  // this is NOT always the same as the affine-grid image number
  // as there can be multiple GRIDS per input image:
  const int ibd = (ib * inSize) / outSize ;

  const int inOffset = ic * (inHeight * inWidth) + ibd*(inHeight * inWidth * inDepth);

  for (int i=0; i < 2; i++) {
    for (int j=0; j< 2; j++) {
      ssx = sx + i;
      ssy = sy + j;
      // again, check for out-of-range:
      if (ssx < 0 || ssx > inHeight-1 || ssy < 0 || ssy > inWidth-1) {continue;}
      const float w = ((1-i)*(1-wx) + i*wx) * ((1-j)*(1-wy) + j*wy);
      outval += w * xin[ssx + ssy * inHeight + inOffset];
    }
  }
  out[flatIndex] = outval;
}

/** Computes the gradient of the input data values
 *  (G_XIN), given the gradients wrt the output (G_Y).
 *
 *  Note: since multiple output pixels can map to the same
 *        input pixel, we have to use ATOMIC-ADD to write
 *        back the gradient value -- this could be a bottleneck
 *        (but perhaps there's no simple way around this).
 *        We assume that there is a thread running for each 
 *        output pixel. **/

// for (float) atomicAdd:
__global__ void bilinearsampler_bwd_data_kernel(float* const g_xin,
                          const float* const g_y,
                          const float* const xin,
                          const float* const grid,
                          const int outHeight,
                          const int outWidth,
                          const int outSize,
                          const int inHeight,
                          const int inWidth,
                          const int inDepth,
                          const int inSize) {

  const int flatIndex = getGlobalIdx_2D_1D();
  const int nOut = outWidth * outHeight * inDepth * outSize;
  // thread boundary check:
  if (flatIndex >= nOut) { return; }

  // extract the indices of the output location 
  // that this thread is computing (ix,iy,ic,ib):  
  int ix = flatIndex; // x index
  int iy = ix / outHeight; // y index
  int ic = iy / outWidth; // channel index
  const int ib = ic / inDepth;  // image index (in the batch)
  ix %= outHeight; 
  iy %= outWidth;
  ic %= inDepth;

  // get the values of source-x and source-y from the
  // correct location in the grid:
  // grid has the size: outHeight x outWidth x 2 x nBatch
  const int igx = ix + iy*outHeight + ib * 2*outHeight*outWidth;
  const int igy = igx + outHeight*outWidth;
  // get the location from the grid:
  // and convert from [-1,1] --> [0,H-1]
  float px = grid[igx];
  float py = grid[igy];
  px = (px + 1.0) / 2.0 * (inHeight-1);
  py = (py + 1.0) / 2.0 * (inWidth-1);
  const int sx = floor(px);
  const int sy = floor(py);
  // pre-emptive check here for out of range:
  if (sx < -1 || sx > inHeight-1 || sy < -1 || sy > inWidth-1) {
    // this output pixel is not linked with any
    // input pixel value => do-nothing
    return;
  }

  // get the interpolation weights:
  const float wx = px - sx;
  const float wy = py - sy;
  // add the weighted sum to the output:
  const float d_y = g_y[flatIndex];
  
  // get the number of input-image from which we get the data:
  // this is NOT always the same as the affine-grid image number
  // as there can be multiple GRIDS per input image:
  const int ibd = (ib * inSize) / outSize ;
  
  const int inOffset = ic * (inHeight * inWidth) + ibd*(inHeight * inWidth * inDepth);

  for (int i=0; i < 2; i++) {
    const int ssx = sx + i;
    if (ssx < 0 || ssx > inHeight-1) {continue;}
    for (int j=0; j< 2; j++) {
      const int ssy = sy + j;
      // again, check for out-of-range:
      if (ssy < 0 || ssy > inWidth-1) {continue;}
      const float w = ((1-i)*(1-wx) + i*wx) * ((1-j)*(1-wy) + j*wy);
      // add the gradient to the output buffer:
      // could be slow due to locking
      atomicAdd( g_xin + ssx+ssy*inHeight+inOffset, w*d_y);
    }
  }
}

/** Computes the gradient of the grid-coordinates
 *  (G_GRID), given the gradients wrt the output (G_Y).
 *
 *  Note: that the gradient wrt a coordinate (gx or gy)
 *        is the sum of gradients from each channel.
 *        Therefore to avoid writing atomic adds,
 *        each thread is reponsible for all channels
 *        correspoding to a given output spatial location
 *        of a given image in the batch. **/
__global__ void  bilinearsampler_bwd_grid_kernel(float* const g_grid,
                          const float* const g_y,
                          const float* const xin,
                          const float* const grid,
                          const int outHeight,
                          const int outWidth,
                          const int outSize,
                          const int inHeight,
                          const int inWidth,
                          const int inDepth,
                          const int inSize) {

  const int flatIndex = getGlobalIdx_2D_1D();
  const int n_out_spatial = outHeight * outWidth;
  const int nTh = n_out_spatial * outSize;
  if (flatIndex >= nTh) {return;}

  // extract the indices of spatial location and image:
  int ix = flatIndex;
  int iy = ix / outHeight;
  const int ib = iy / outWidth;
  ix %= outHeight;
  iy %= outWidth;

  // get the values of source-x and source-y from the
  // correct location in the grid:
  const int igx = ix + iy*outHeight + ib * 2*n_out_spatial;
  const int igy = igx + n_out_spatial;
  float px = grid[igx];
  float py = grid[igy];
  const float scale_x_geom = (inHeight-1)/2.0;
  const float scale_y_geom = (inWidth-1)/2.0;
  px = (px + 1.0) * scale_x_geom;
  py = (py + 1.0) * scale_y_geom;
  const int sx = floor(px);
  const int sy = floor(py);

  // pre-emptive check here for out of range:
  if (sx < -1 || sx > inHeight-1 || sy < -1 || sy > inWidth-1) {
    // this is out of range, this means that these coords
    // do not have any impact on the output => we back-prop 0:
    g_grid[igx] = 0;
    g_grid[igy] = 0;
    return;
  }

  // get the interpolation weights:
  const float wx = px - sx;
  const float wy = py - sy;
  
  // initialize the accumulators for contributions
  // from each channel to zero:
  float g_tot_x = 0, g_tot_y = 0;

  // get the number of input-image from which we get the data:
  // this is NOT always the same as the affine-grid image number
  // as there can be multiple GRIDS per input image:
  const int ibd = (ib * inSize) / outSize ;

  const int n_in_spatial = inHeight * inWidth;
  const int in_batch_offset = ibd*(n_in_spatial * inDepth);
  const int out_batch_offset = ix + iy*outHeight + ib*n_out_spatial*inDepth;

  // iterate over the 4-neighborhood induced by each
  // x,y coordinate pair in the grid:
  for (int i=0; i < 2; i++) {
    const int ssx = sx + i;
    if (ssx < 0 || ssx > inHeight-1) {continue;}
    for (int j=0; j< 2; j++) {
      const int ssy = sy + j;
      if (ssy < 0 || ssy > inWidth-1) {continue;}

      const float w_x =  (2*i-1) * ((1-j)*(1-wy) + j*wy); // +-1 * wy
      const float w_y =  (2*j-1) * ((1-i)*(1-wx) + i*wx); // +-1 * wx
      const int in_offset = ssx + ssy*inHeight + in_batch_offset;
      // iterate over each channel, accumulating the gradients:
      for (int ic=0; ic < inDepth; ic++) {
        const int out_offset =  ic * n_out_spatial + out_batch_offset;
        const float d_xin = g_y[out_offset] * xin[in_offset + ic*n_in_spatial];
        g_tot_x += w_x * d_xin;
        g_tot_y += w_y * d_xin;
      }
    }
  }
  g_grid[igx] = scale_x_geom * g_tot_x;
  g_grid[igy] = scale_y_geom * g_tot_y;
}

/** get the number of threads (1D) and blocks (2D). **/
vl::Error get_launch_params(const int& N, int& nTh, int& nGx, int& nGy) {
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

/** INTERFACE to the kernels:
*  Sets up the threads and blocks and run the kernels. */
vl::Error
vl::impl::bilinearsampler::forward (float* output,
                                    float const* data,
                                    float const* grid,
                                    size_t outHeight, 
                                    size_t outWidth,
                                    size_t outSize,
                                    size_t inHeight,
                                    size_t inWidth,
                                    size_t inChannels,
                                    size_t inSize) {
  
  // setup and launch the kernel:
  const int outVolume = outHeight * outWidth * inChannels * outSize;
  int nTh, nGx, nGy;
  vl::Error volume_ok = get_launch_params(outVolume, nTh, nGx, nGy);
  if (volume_ok != vl::vlSuccess) { return volume_ok;}

  dim3  gridDim(nGx,nGy); // grid-dimensions
  bilinearsampler_fwd_kernel <<< gridDim, nTh >>> (output, data, grid,
                                                   outHeight, outWidth,
                                                   outSize,
                                                   inHeight, inWidth,
                                                   inChannels,
                                                   inSize);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

vl::Error
vl::impl::bilinearsampler::backward (float* derData,
                                     float* derGrid,
                                     float const* data,
                                     float const* grid,
                                     float const* derOutput,
                                     size_t outHeight, 
                                     size_t outWidth,
                                     size_t outSize,
                                     size_t inHeight,
                                     size_t inWidth,
                                     size_t inChannels,
                                     size_t inSize) {

  // setup and launch the kernel for DER-DATA:
  const int outVolume = outHeight * outWidth * inChannels * outSize;
  int nTh, nGx, nGy;
  vl::Error volume_ok = get_launch_params(outVolume, nTh, nGx, nGy);
  if (volume_ok != vl::vlSuccess) { return volume_ok;}

  dim3  gridDim(nGx,nGy); // grid-dimensions
  bilinearsampler_bwd_data_kernel <<< gridDim, nTh >>> (derData, derOutput,
                                                        data, grid,
                                                        outHeight, outWidth,
                                                        outSize,
                                                        inHeight, inWidth,
                                                        inChannels, inSize);

  // check if we errored, if yes abort:
  cudaError_t status = cudaPeekAtLastError() ;
  if (status != cudaSuccess) { return vl::vlErrorCuda; }

  // setup and launch kernel for DER-GRID:
  const int outN = outHeight * outWidth * outSize;
  volume_ok = get_launch_params(outN, nTh, nGx, nGy);
  if (volume_ok != vl::vlSuccess) { return volume_ok;}

  gridDim.x = nGx; gridDim.y = nGy; // grid-dimensions
  bilinearsampler_bwd_grid_kernel <<< gridDim, nTh >>> (derGrid, derOutput,
                                                       data, grid,
                                                       outHeight, outWidth,
                                                       outSize,
                                                       inHeight, inWidth,
                                                       inChannels, inSize );
  // catch any errors:
  status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
