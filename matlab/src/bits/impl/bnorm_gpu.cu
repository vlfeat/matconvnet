// @file   bnorm_gpu.cu
// @brief  Batch normalization implementation (GPU)
// @author Sebastien Ehrhardt

/*
Copyright (C) 2015 Sebastien Ehrhardt.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bnorm.hpp"
#include "../datacu.hpp"
#include "blashelper.hpp"
#include <assert.h>
#include <float.h>
#include <stdint.h>

// MSB_WARP = log2(WARP_SIZE)
#define WARP_SIZE 32
#define MSB_WARP 5


// macro function
#define min(a,b) (a > b ? b : a);
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                                         Helpers	*/
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */


static inline int getBlockSize(int dataSize)
{
  int blockSize = VL_CUDA_NUM_THREADS / 2 ;
  if (dataSize < blockSize) {
    unsigned int numWarps = dataSize / WARP_SIZE ;
    if (numWarps < 4) {
      blockSize = 2 * WARP_SIZE ;
    }
    else if (numWarps < 8) {
      blockSize = 4 * WARP_SIZE ;
    }
    else {
      blockSize = 8 * WARP_SIZE ;
    }
  }
  return blockSize ;
}

// The bockReduce function(s) computes the sum of the elements of the
// array mdata[] (and sdata, rdata, tdata):
//
// mdata[0] <- mdata[0] + mdata[1] + ... + mdata[blockSize-1]
//
// blockSize is a power of two.
//
// When the reduction involves a single warp of 32 threads further
// optimisation kick in. In fact, such threads work synchronously in the warp, and explicit syncrhonisation is not needed anymore.

__forceinline__ __device__ void warpReduce(volatile float * mdata,
                                           unsigned int tid,
                                           unsigned int blockSize)
{
  if (blockSize >=  64) { mdata[tid] += mdata[tid + 32]; } // mdata[0:31] = mdata[0:31] + mdata[32:63]
  if (blockSize >=  32) { mdata[tid] += mdata[tid + 16]; } // mdata[0:15] = mdata[0:15] + mdata[16:31]
  if (blockSize >=  16) { mdata[tid] += mdata[tid +  8]; } // mdata[0:7] = mdata[0:7] + mdata[7:15]
  if (blockSize >=   8) { mdata[tid] += mdata[tid +  4]; } // mdata[0:3] = mdata[0:3] + mdata[4:7]
  if (blockSize >=   4) { mdata[tid] += mdata[tid +  2]; } // mdata[0:1] = mdata[0:1] + mdata[2:3]
  if (blockSize >=   2) { mdata[tid] += mdata[tid +  1]; } // mdata[0] = mdata[0] + mdata[1]
}

__forceinline__ __device__ void blockReduce(volatile float * mdata,
                                            unsigned int tid,
                                            unsigned int blockSize,
                                            unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >=512) { if (tid < 512) { mdata[tid] += mdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512  && maxDataSize + WARP_SIZE >=256) { if (tid < 256) { mdata[tid] += mdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256  && maxDataSize + WARP_SIZE >=128) { if (tid < 128) { mdata[tid] += mdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128  && maxDataSize + WARP_SIZE >=64 ) { if (tid <  64) { mdata[tid] += mdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) {
    warpReduce(mdata, tid, blockSize);
  }
}

__forceinline__ __device__ void warpReduce2(volatile float * sdata, volatile float * mdata, unsigned int tid, unsigned int blockSize)
{
  if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; }
  if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; }
  if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; }
  if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; }
  if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; }
  if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; }
}

__forceinline__ __device__ void blockReduce2(volatile float * mdata,
                                             volatile float * sdata,
                                             unsigned int tid,
                                             unsigned int blockSize,
                                             unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >=512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512  && maxDataSize + WARP_SIZE >=256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256  && maxDataSize + WARP_SIZE >=128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128  && maxDataSize + WARP_SIZE >=64)  { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) {
    warpReduce2(sdata, mdata, tid, blockSize);
  }
}

__forceinline__ __device__ void warpReduce4(volatile float * sdata,
                                            volatile float * mdata,
                                            volatile float * rdata,
                                            volatile float * tdata,
                                            unsigned int tid,
                                            unsigned int blockSize)
{
  if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; rdata[tid] += rdata[tid + 32]; tdata[tid] += tdata[tid + 32];}
  if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; rdata[tid] += rdata[tid + 16]; tdata[tid] += tdata[tid + 16];}
  if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; rdata[tid] += rdata[tid +  8]; tdata[tid] += tdata[tid +  8];}
  if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; rdata[tid] += rdata[tid +  4]; tdata[tid] += tdata[tid +  4];}
  if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; rdata[tid] += rdata[tid +  2]; tdata[tid] += tdata[tid +  2];}
  if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; rdata[tid] += rdata[tid +  1]; tdata[tid] += tdata[tid +  1];}
}

__forceinline__ __device__ void blockReduce4(volatile float * sdata,
                                             volatile float * mdata,
                                             volatile float * rdata,
                                             volatile float * tdata,
                                             unsigned int tid,
                                             unsigned int blockSize,
                                             unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >= 512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; rdata[tid] += rdata[tid + 512]; tdata[tid] += tdata[tid + 512];} __syncthreads(); }
  if (blockSize >= 512 && maxDataSize + WARP_SIZE >= 256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; rdata[tid] += rdata[tid + 256]; tdata[tid] += tdata[tid + 256];} __syncthreads(); }
  if (blockSize >= 256 && maxDataSize + WARP_SIZE >= 128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; rdata[tid] += rdata[tid + 128]; tdata[tid] += tdata[tid + 128];} __syncthreads(); }
  if (blockSize >= 128 && maxDataSize + WARP_SIZE >= 64) { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  rdata[tid] += rdata[tid + 64]; tdata[tid] += tdata[tid + 64];} __syncthreads(); }
  if (tid < 32) {
    warpReduce4(sdata, mdata, rdata, tdata, tid, blockSize);
  }
}

/*
 In the following we often need to use blocks of threads to sum over
 data which is not necessarily naturally aligned with thread blocks or even thread warps.

 The trick is to look at the block as a jumping window, sliding it over the memory
 that needs to be summed, but always aligned at natural block boundaries. This means
 that occasionally blocks will only be partially filled with useful memory:

    +-------+ +-------+           +-------+ +-------+      aligned blocks (with two warps each)
    |   :   | |   :   |           |   :   | |   :   |      covering the data
    +-------+ +-------+           +-------+ +-------+
      +-------------+             +-------------+          data to sum

+-------------------------------------------------------->
              increasing memory addresses


 This pattern is repreated several times in the code below.
 */


// Get largest memory address that is aligned to a warp worth of float
// and smaller than x.

__forceinline__ __device__ uintptr_t getBlockBeginning(void const * x)
{
  return (uintptr_t)(x) & (~((uintptr_t)(WARP_SIZE*sizeof(float)) - 1)) ;
}

// Use the current block of thread to sum over a given column of a matrix. The selected
// column is given by the thread block index in the block grid.

__forceinline__ __device__ float matrixSumHelper(float const * matrix, int numRows)
{
  // One thread block per column to sum
  extern __shared__ float scratch [] ;
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  int blockSize = blockDim.x ;

  scratch[tid] = 0 ;

  float const * columnBegin = matrix + column * numRows ;
  float const * columnEnd = columnBegin + numRows ;
  float const * block = (float const*) getBlockBeginning(columnBegin) + tid ;
  while (block < columnEnd) {
    if (block >= columnBegin) {
      scratch[tid] += *block ;
    }
    block += blockSize ;
  }

  // Now we have a block worth of partial sums for the column
  // Finish by reducing and saving
  blockReduce(scratch, tid, blockSize, numRows) ;

  return scratch[0] ;
}

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                                    bnorm_forward	*/
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */


__global__ void divide(float * mu,
                       float * sigma,
                       float mass,
                       unsigned int n)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx < n){
    float mean = mu[idx]/mass;
    mu[idx] = mean;
    sigma[idx] = sigma[idx]/mass-mean*mean;
  }
}

// The kernel accumulates means and variances for the data.
// Each block of thread sums over one or more data planes, resulting
// in an array accumulator[] of dimension numChunks x 2*numChannels.
//
// If each thread block scans all the images, then numChunks = 1.
// However, for efficiency different thread blocks do different
// subset of images, resulting in numChunks partial results to be integrated
// later.
//
// The first part accumulator[:,0:numChannels-1] stores the data for the mean
// and the second part accumulator[:,numChannels,2*numChannels-1] the data
// for the sigmas.

__global__ void computePartialMuSigma(float * accumulator,
                                      float const * data,
                                      int planeArea,
                                      int numPlanes,
                                      int numChannels,
                                      int numChunks)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;
  extern __shared__ float s[] ;
  float * mdata = s ;
  float * sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    float const * planeBegin = data + plane * planeArea ;
    float const * planeEnd = planeBegin + planeArea ;
    float const * block = (float const*) getBlockBeginning(planeBegin) + tid ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        float x = *block ;
        mdata[tid] += x ;
        sdata[tid] += x * x ;
      }
      block += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce2(sdata, mdata, tid, blockSize, planeArea) ;

  if (tid == 0) {
    int chunk = blockIdx.x / numChannels ;
    int i = chunk + channel * numChunks ;
    accumulator[i] = mdata[0];
    accumulator[i + gridDim.x] = sdata[0];
  }
}

__global__ void reduceMuSigma(float * accumulator,
                              float const * matrix,
                              int numRows)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  float x = matrixSumHelper(matrix, numRows) ;
  if (tid == 0) {
    accumulator[column] = x ;
  }
}

__global__ void normalize(float * outputData,
                          float const * data,
                          float const * means,
                          float const * sigmas,
                          float const * multipliers,
                          float const * biases,
                          int planeArea,
                          int numPlanes,
                          int numChannels,
                          float epsilon)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  // Not optimized for compute capability < 1.2
  float mean = means[channel];
  float sigma = sigmas[channel];
  float multiplier = multipliers[channel];
  float bias = biases[channel];
  float coefficient = multiplier * rsqrt(sigma + epsilon) ;

  while (plane < numPlanes) {
    float const * planeBegin = data + plane * planeArea ;
    float const * planeEnd = planeBegin + planeArea ;
    float const * block = (float const*) getBlockBeginning(planeBegin) + tid ;
    float * oblock = outputData + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        *oblock = coefficient * (*block - mean) + bias ;
      }
      block += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}

template<> vl::Error
vl::impl::bnorm_forward<vl::GPU, float>(Context& context,
                                        float* output,
                                        float const* data,
                                        float const* multipliers,
                                        float const* biases,
                                        int height, int width, int depth, int size,
                                        float epsilon)
{

/*

 The data is organised in SIZE images, each of which is composed of DEPTH
 planes. The goal is to compute the mean and std of the features in each
 plane. In the follwing diagram, planes are enumerated from left to right
 and top to bottom, listing first all the plane for one image (a row) and then
 subsequent images (in different rows).

     +-------+   +-------+   +-------+   +-------+                             
     |plane 1|   |p 2    |   |p 3    |   |p 4    |  numPlanes = 12             
     |ch 1   |   |c 2    |   |c 3    |   |c 4    |  depth = 4                  
     |image 1|   |i 1    |   |i 1    |   |i 1    |  planeArea = 28
 +---+block 1|   |b 2    |   |b 3    |   |b 4    |  planeStride = gridSize = 8
 |   +-------+   +-------+   +-------+   +-------+                             
 |                                                                             
 |   +-------+   +-------+   +-------+   +-------+                             
 |   |p 5    |   |p 6    |   |p 7    |   |p 8    |                             
 |   |c 1    |   |c 2    |   |c 3    |   |c 4    |                             
 |   |i 2    |   |i 2    |   |i 2    |   |i 2    |                             
 |   |b 5    |   |b 6    |   |b 7    |   |b 8    |                             
 |   +-------+   +-------+   +-------+   +-------+                             
 |                                                                             
 |   +-------+   +-------+   +-------+   +-------+                             
 |   |p 9    |   |p 10   |   |p 11   |   |p 12   |                             
 |   |c 1    |   |c 2    |   |c 3    |   |c 4    |                             
 |   |i 3    |   |i 3    |   |i 3    |   |i 3    |
 +-->+b 1    |   |b 2    |   |b 3    |   |b 4    |
     +-------+   +-------+   +-------+   +-------+                             


 We create gridSize thread blocks. Each block is assigned to sum
 over a successive plane in the order above. Since there may be less blocks
 than planes overall, these warp around (in the example, thread block 1
 integrates planes 1 and planes 9).

 */

  float *mean ;
  float *sigma ;
  cudaError_t status;
  vl::Device type = GPU;
  unsigned int planeArea = height * width ;
  unsigned int numPlanes = depth * size ;
  unsigned int blockSize = getBlockSize(planeArea) ;

  // Try allocating one block for each plane. However, if
  // this corresponds to too many blocks, reduce the number,
  // still making sure that the number of blocksis a multiple of
  // DEPTH. The latter is needed so that a block always sums
  // features belonging to the same channel,
  // even across different images.
  unsigned int row = 1 ;
  unsigned int gridSize =  depth ;

  // Avoid thread overload : a thread will execute less than ten thousand operation
  /*if (planeArea*size > 10000*blockSize) {
    row = min((depth*planeArea*size)/(9999*blockSize)+1,size) ;
    // gridSize limit
    if(depth >= 65536){
      row = 1;
    }
    else if (depth*row > 65536) {
      row = 65536/depth + 1 ;
    }
    gridSize = row * depth ;
  }*/

  if (gridSize != depth){

    // Get intermediate buffers
    unsigned int fin1 = (gridSize%WARP_SIZE==0) ? gridSize : WARP_SIZE*((gridSize>>MSB_WARP)+1);
    float * intermediateOutput = (float*) context.getWorkspace(type, (gridSize+fin1+2*depth) * sizeof(float)) ;
    mean = intermediateOutput + gridSize+fin1;
    sigma = mean + depth;

    // Compute mean and variance at the same time
    computePartialMuSigma <<<gridSize, blockSize, 2*blockSize*sizeof(float)>>>
    (intermediateOutput,
     data,
     planeArea,
     numPlanes,
     depth,
     row) ;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::vlErrorCuda ;

    int blockSizeSum = getBlockSize(row) ;
    reduceMuSigma <<<2*depth,blockSizeSum,blockSizeSum*sizeof(float)>>>
    (mean, intermediateOutput, row) ;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::vlErrorCuda ;

  } else {
    mean = (float*) context.getWorkspace(type, 2*depth * sizeof(float)) ;
    sigma = mean + depth;

    computePartialMuSigma<<<gridSize, blockSize, 2*blockSize*sizeof(float)>>>
    (mean,
     data,
     planeArea,
     numPlanes,
     depth,
     1) ;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::vlErrorCuda ;
  }

  unsigned int mass = planeArea*size;
  divide <<<divideUpwards(depth,blockSize),blockSize>>>
  (mean, mean+depth, (float)mass, depth);

  normalize <<<gridSize, blockSize>>>
  (output, data, mean, sigma, multipliers, biases,
   planeArea,
   numPlanes,
   depth,
   epsilon) ;

  status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                                   bnorm_backward */
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */

__global__ void divideSigma(float * dzdg,
                            float * dzdb,
                            float * mu,
                            float * sigma,
                            float epsilon,
                            float mass,
                            unsigned int n
                            )
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n){
    float mean = mu[idx]/mass;
    mu[idx] = mean;
    sigma[idx] = sigma[idx]/mass-mean*mean;
    dzdg[idx] = (dzdg[idx]-mean*dzdb[idx])/sqrt(sigma[idx]+epsilon);
  }
}

__global__ void computePartialMuSigmaDer(float * buffer1,
                                         float * buffer2,
                                         float * buffer3,
                                         float const * data,
                                         float const * derOutput,
                                         int planeArea,
                                         int numPlanes,
                                         int numChannels,
                                         int numChunks)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;
  extern __shared__ float s[] ;
  float * mdata = s ;
  float * sdata = mdata + blockSize ;
  float * rdata = sdata + blockSize ;
  float * tdata = rdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;
  rdata[tid] = 0 ;
  tdata[tid] = 0 ;

  while (plane < numPlanes) {
    float const * planeBegin = data + plane * planeArea ;
    float const * planeEnd = planeBegin + planeArea ;
    float const * block = (float const*) getBlockBeginning(planeBegin) + tid ;
    float const * dblock = derOutput + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        float x = *block ;
        float dy = *dblock ;
        mdata[tid] += x * dy ;
        sdata[tid] += dy ;
        rdata[tid] += x * x ;
        tdata[tid] += x ;
      }
      block += blockSize ;
      dblock += blockSize ;
    }
    plane += planeStride ;
  }

  // Nothing to optimize here

  blockReduce4(sdata, mdata, rdata, tdata, tid, blockSize, planeArea);
  if (tid == 0) {
    if (numChannels == gridDim.x) {
      // Final output ready
      buffer1[blockIdx.x] = mdata[0];
      buffer2[blockIdx.x] = sdata[0];
      buffer3[blockIdx.x] = tdata[0];
      buffer3[blockIdx.x+numChannels] = rdata[0];
    } else {
      // Partially accumulated outut
      int chunk = blockIdx.x / numChannels ;
      int i = chunk + channel * numChunks ;
      buffer1[i] = mdata[0]; // derMultipliers
      buffer1[i + gridDim.x] = sdata[0]; // derBiases
      buffer1[i + 2*gridDim.x] = tdata[0]; // means
      buffer1[i + 3*gridDim.x] = rdata[0]; // sigmas
    }
  }
}

__global__ void reduceMuSigmaDer(float * accumulator,
                                 float * derMultipliers,
                                 float * derBiases,
                                 float const * matrix,
                                 int numRows,
                                 int numChannels)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  float x = matrixSumHelper(matrix, numRows) ;
  if (tid == 0) {
    // Recall that the matrix stores in order [derMultipliers derBiases means sigmas]
    // containing four types of data
    int type = column / numChannels ;
    int channel = column % numChannels ;

    if (type == 0) {
      derMultipliers[channel] = x ;
    }
    else if (type == 1) {
      derBiases[channel] = x ;
    }
    else if (type == 2) {
      accumulator[channel] = x ;
    }
    else {
      accumulator[channel + numChannels] = x ;
    }
  }
}

__global__ void normalizeBackward(float * derData,
                                  float const * data,
                                  float const * derOutput,
                                  float const * means,
                                  float const * sigmas,
                                  float const * multipliers,
                                  float const * derBiases,
                                  float const * derMultipliers,
                                  int planeArea,
                                  int numPlanes,
                                  int numChannels,
                                  float epsilon,
                                  float mass)

{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  // Not optimized for compute capability < 1.2
  float mu = means[channel];
  float sigma2 = sigmas[channel] + epsilon;
  float multiplier = multipliers[channel] ;
  float muz = derBiases[channel] / mass;
  float derMultiplier = derMultipliers[channel];
  float G1 = multiplier * rsqrt(sigma2);
  float G2 = (multiplier * derMultiplier) / (sigma2 * mass);

  while (plane < numPlanes) {
    float const * planeBegin = data + plane * planeArea ;
    float const * planeEnd = planeBegin + planeArea ;
    float const * block = (float const*) getBlockBeginning(planeBegin) + tid ;
    float const * dblock = derOutput + (block - data) ;
    float * oblock = derData + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        *oblock = G1 * (*dblock - muz) - G2 * (*block - mu);
      }
      block += blockSize ;
      dblock += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}

template<> vl::Error
vl::impl::bnorm_backward<vl::GPU, float>(Context& context,
                                         float* derData,
                                         float* derMultipliers,
                                         float* derBiases,
                                         float const* data,
                                         float const* multipliers,
                                         float const* biases,
                                         float const* derOutput,
                                         int height, int width, int depth, int size,
                                         float epsilon)
{
  vl::Device type = GPU;
  float *intermediateOutput;
  float *mean ;
  float *sigma;
  cudaError_t status;
  unsigned int planeArea = height * width ;
  unsigned int numPlanes = depth * size ;
  unsigned int blockSize = getBlockSize(planeArea) ;

  unsigned int row = 1 ;
  unsigned int gridSize = depth ;

  // Avoid thread overload : a thread will execute less than ten thousand operation
  /*if (planeArea*size > 10000*blockSize) {
    row = min((depth*planeArea*size)/(9999*blockSize)+1,size) ;
    // gridSize limit
    if(depth >= 65536){
      row = 1;
    }
    else if (depth*row > 65536) {
      row = 65536/depth + 1 ;
    }
    gridSize = row * depth ;
  }*/

  if(gridSize != depth){

    // Get intermediate buffers
    unsigned int fin1 = (gridSize%WARP_SIZE==0) ? gridSize : WARP_SIZE*((gridSize>>MSB_WARP)+1);
    // Might be optimize here to get coalescent access
    intermediateOutput = (float*) context.getWorkspace(type, (3*gridSize+fin1+2*depth) * sizeof(float)) ;
    mean = intermediateOutput + fin1 + 3*gridSize;
    sigma = mean + depth;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::vlErrorCuda ;

    // Mean, variance, derMultipliers and derBiases computation
    computePartialMuSigmaDer<<<gridSize, blockSize, 4*blockSize*sizeof(float)>>>
    (intermediateOutput,
     NULL,
     NULL,
     data,
     derOutput,
     planeArea,
     numPlanes,
     depth,
     row) ;

    int blockSizeSum = getBlockSize(row) ;
    reduceMuSigmaDer <<<4*depth,blockSizeSum,blockSizeSum*sizeof(float)>>>
    (mean, derMultipliers, derBiases,
     intermediateOutput, row, depth) ;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::vlErrorCuda ;
  } else {
    mean = (float*) context.getWorkspace(type, (2*depth) * sizeof(float)) ;
    sigma = mean + depth;

    computePartialMuSigmaDer <<<gridSize, blockSize, 4*blockSize*sizeof(float)>>>
    (derMultipliers,
     derBiases,
     mean,
     data,
     derOutput,
     planeArea,
     numPlanes,
     depth,
     1) ;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::vlErrorCuda ;
  }

  unsigned int mass = planeArea*size;
  divideSigma<<<divideUpwards(depth,blockSize),blockSize>>>
  (derMultipliers, derBiases, mean,sigma,epsilon,(float)mass,depth);

  // Compute output
  normalizeBackward <<<gridSize, blockSize>>>
  (derData,
   data, derOutput, mean, sigma,
   multipliers, derBiases, derMultipliers,
   planeArea, numPlanes, depth,
   epsilon, (float)mass) ;

  status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
