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
#include "../datamex.hpp"
#include "../datacu.hpp"
#include "blashelper.hpp"
#include <assert.h>
#include <float.h>

#define WARP_SIZE 32
#define MSB_WARP 5

/* ---------------------------------------------------------------- */
/*                                              bnorm_forward	      */
/* ---------------------------------------------------------------- */

__device__ void warpReduce(volatile float * mdata, unsigned int tid, unsigned int blockSize) {
  if (blockSize >=  64) {  mdata[tid] += mdata[tid + 32]; }
  if (blockSize >=  32) {  mdata[tid] += mdata[tid + 16]; }
  if (blockSize >=  16) {  mdata[tid] += mdata[tid +  8]; }
  if (blockSize >=   8) {  mdata[tid] += mdata[tid +  4]; }
  if (blockSize >=   4) {  mdata[tid] += mdata[tid +  2]; }
  if (blockSize >=   2) {  mdata[tid] += mdata[tid +  1]; }
}

//template <unsigned int blockSize>
__device__ void warpDoubleReduce(volatile float * sdata, volatile float * mdata, unsigned int tid, unsigned int blockSize) {
  if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; }
  if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; }
  if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; }
  if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; }
  if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; }
  if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; }
}

__device__ void warp4Reduce(volatile float * sdata, volatile float * mdata, volatile float * rdata, volatile float * tdata, unsigned int tid, unsigned int blockSize) {
  if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; rdata[tid] += rdata[tid + 32]; tdata[tid] += tdata[tid + 32];}
  if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; rdata[tid] += rdata[tid + 16]; tdata[tid] += tdata[tid + 16];}
  if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; rdata[tid] += rdata[tid +  8]; tdata[tid] += tdata[tid +  8];}
  if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; rdata[tid] += rdata[tid +  4]; tdata[tid] += tdata[tid +  4];}
  if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; rdata[tid] += rdata[tid +  2]; tdata[tid] += tdata[tid +  2];}
  if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; rdata[tid] += rdata[tid +  1]; tdata[tid] += tdata[tid +  1];}
}


__device__ void accessDataSum(
		float * g_idata,
		volatile float * mdata,
    unsigned int row,//row
    unsigned int quotient, // row/blockSize
    unsigned int remain, // row%WARP_SIZE
    unsigned int dec,
    unsigned int n,
    unsigned int blockSize,
    unsigned int blockidx,
    unsigned int tid)
{
  unsigned int y = blockidx*(row);
  unsigned int r = y - (y>>MSB_WARP)*WARP_SIZE;//y%WARP_SIZE
  unsigned int i = y + tid -r;// 2* can be added to blockSize, will be modified later
  unsigned int iter = 0;

  if (row >= blockSize) {
    if (remain == 0) {
      // if r==0 all memory access are coalesced
      r = row-quotient*blockSize;//row%blockSize
      if( r==0){
        while (iter < quotient){
          mdata[tid] += g_idata[i];
          i += blockSize;
          iter++;
        }
      } else {
        while (iter < quotient+1){
          if (iter == quotient) {
            // The last warps may have divergent kernels
            if(tid<r) mdata[tid] += g_idata[i];
            iter++;
          } else {
            mdata[tid] += g_idata[i];
            i += blockSize;
            iter++;
          }
        }
      }
    } else {
      // Memory access are made coalesced this imply that some kernel will be waiting while some will be working
      if(r>dec){
        while (iter < quotient+1){
          // Make all memory access coalesced
          if (iter ==0) {
            if (tid >= r){ mdata[tid] += g_idata[i]; }
            else if (tid < r-dec){ mdata[tid] += g_idata[i+row+dec]; }
            i += blockSize;
            iter++;
          }
          // if blockSize is too small then can be quotient+1 instead of quotient (arrive when r > blockSize/2 so never in practice)
          else if (iter == quotient) {
            if(tid < blockSize - dec + r) mdata[tid] += g_idata[i];
            iter++;
          } else{
            mdata[tid] += g_idata[i];
            i += blockSize;
            iter++;
          }
        }
      } else{
        while (iter < quotient+1){
          // Make all memory access coalesced
          if (iter ==0) {
            if (tid >= r) mdata[tid] += g_idata[i];
            i += blockSize;
            iter++;
          }
          // if blockSize is too small then can be quotient+1 instead of quotient (arrive when r > blockSize/2 so never in practice)
          else if (iter == quotient) {
            if(tid < blockSize - dec + r) mdata[tid] += g_idata[i];
            iter++;
          } else{
            mdata[tid] += g_idata[i];
            i += blockSize;
            iter++;
          }
        }
      }
    }
  } else {
    // If row is too small. Highly suboptimal
    if (remain == 0) {
      // The last warps may have divergent kernels
      if(tid<row) mdata[tid] += g_idata[i];
    } else {
      if(tid>=r && tid < r+row) { mdata[tid] += g_idata[i]; }
      else if(r+row > blockSize && tid < row +r -blockSize) { mdata[tid] += g_idata[i+blockSize]; }
    }
  }
  
  // Nothing to optimize here
  __syncthreads();

  if (blockSize >= 1024 && row + WARP_SIZE >=512) { if (tid < 512) {  mdata[tid] += mdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512  && row + WARP_SIZE >=256) { if (tid < 256) {  mdata[tid] += mdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256  && row + WARP_SIZE >=128) { if (tid < 128) {  mdata[tid] += mdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128  && row + WARP_SIZE >=64) { if (tid <  64) { mdata[tid] += mdata[tid + 64];  } __syncthreads(); }
}

__global__ void computeMatrixSum(
    float * g_idata,
    float * g_ires,
    unsigned int row,//row
    unsigned int quotient, // row/blockSize
    unsigned int remain, // row%WARP_SIZE
    unsigned int dec,
    unsigned int n)
{
  extern __shared__ float s[];
  float *mdata = s;

  unsigned int tid = threadIdx.x;
  unsigned int blockidx = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  mdata[tid] = 0;

  accessDataSum(g_idata, mdata, row, quotient, remain, dec, n, blockSize, blockidx, tid);

  if (tid < 32) warpReduce(mdata, tid, blockSize);
  if (tid == 0) g_ires[blockidx] = mdata[0];
}

__global__ void divide(
    float * mu,
    float * sigma,
    float mass,
    unsigned int n )
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx <n){
    float mean = mu[idx]/mass;
    mu[idx] = mean;
    sigma[idx] = sigma[idx]/mass-mean*mean;
  }
}

//template <unsigned int blockSize>
__global__ void computeMuSigma(
    float const * g_idata,
    float * g_odata1,
    unsigned int depth,
    unsigned int row,
    unsigned int WH,//width*height
    unsigned int quotient, // WH/blockSize
    unsigned int remain, // WH%WARP_SIZE
    unsigned int j,
    unsigned int r1,
    unsigned int dec,
    unsigned int n)
{
  extern __shared__ float s[];
  float *mdata = s;
  unsigned int tid = threadIdx.x;
  unsigned int blockidx = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  float *sdata = (float*)&mdata[blockSize];

  unsigned int y = blockidx*(WH);
  unsigned int r = y - (y>>MSB_WARP)*WARP_SIZE;//y%WARP_SIZE
  unsigned int i = y + tid -r;// 2* can be added to blockSize, will be modified later
  unsigned int iter = 0;
  sdata[tid] = 0;
  mdata[tid] = 0;

  if (WH >= blockSize) {
    if (remain == 0) {
      // if r==0 all memory access are coalesced
      r = WH-quotient*blockSize;//WH%blockSize
      if( r==0){
        while (y < n){
          if (iter == quotient-1) {
            mdata[tid] += g_idata[i];
            sdata[tid] += g_idata[i]*g_idata[i];
            y += j;
            i = y +tid;
            iter = 0;
          } else {
            mdata[tid] += g_idata[i];
            sdata[tid] += g_idata[i]*g_idata[i];
            i += blockSize;
            iter++;
          }
        }
      } else {
        while (y < n){
          if (iter == quotient) {
            // The last warps may have divergent kernels
            if(tid<r) {
              mdata[tid] += g_idata[i];
              sdata[tid] += g_idata[i]*g_idata[i];
            }
            y += j;
            i = y+tid;
            iter = 0;
          } else {
            mdata[tid] += g_idata[i];
            sdata[tid] += g_idata[i]*g_idata[i];
            i += blockSize;
            iter++;
          }
        }
      }
    } else {
      // Memory access are made coalesced this imply that some kernel will be waiting while some will be working
      while (y < n){
        // Make all memory access coalesced
        if (iter ==0) {
          if (tid >= r){
            mdata[tid] += g_idata[i];
            sdata[tid] += g_idata[i]*g_idata[i];
          }
          else if(r > dec && tid < r-dec ) {
            mdata[tid] += g_idata[i+WH+dec];
            sdata[tid] += g_idata[i+WH+dec]*g_idata[i+WH+dec];
          }
          i += blockSize;
          iter++;
        }
        // if blockSize is too small then can be quotient+1 instead of quotient (arrive when r > blockSize/2 so never in practice)
        else if (iter == quotient) {
          if(tid < blockSize - dec + r){
            mdata[tid] += g_idata[i];
            sdata[tid] += g_idata[i]*g_idata[i];
          }
          r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
          y += j;
          i = y+tid-r;
          iter =0;
        } else{
          mdata[tid] += g_idata[i];
          sdata[tid] += g_idata[i]*g_idata[i];
          i += blockSize;
          iter++;
        }
      }
    }
  } else {
    // If WH is too small. Highly suboptimal
    if (remain == 0) {
      while(y < n){
        // The last warps may have divergent kernels
        if(tid<WH) {
          mdata[tid] += g_idata[i];
          sdata[tid] += g_idata[i]*g_idata[i];
        }
        y += j;
        i = y+tid;
      }
    } else {
      while (y < n){
        if(tid>=r && tid < r+WH){
          mdata[tid] += g_idata[i];
          sdata[tid] += g_idata[i]*g_idata[i];
        }
        else if( WH +r > blockSize && tid < WH +r -blockSize) {
          mdata[tid] += g_idata[i+blockSize];
          sdata[tid] += g_idata[i+blockSize]*g_idata[i+blockSize];
        }
        r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
        y += j;
        i = y+tid-r;
      }
    }
  }

  // Nothing to optimize here
  __syncthreads();

  if (blockSize >= 1024 && WH + WARP_SIZE >=512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512  && WH + WARP_SIZE >=256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256  && WH + WARP_SIZE >=128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128  && WH + WARP_SIZE >=64) { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  } __syncthreads(); }

  if (tid < 32) warpDoubleReduce(sdata, mdata, tid, blockSize);
  if (tid == 0) {
    i = (blockIdx.x/depth)+(blockIdx.x%depth)*row;
    g_odata1[i] = mdata[0];
    g_odata1[i + gridDim.x] = sdata[0];
  }
}

//template <unsigned int blockSize>
__global__ void normalize(
    float const * g_idata,
    float * g_imean,
    float * g_isigma,
    float * g_ig,
    float * g_ib,
    float * g_odata,
    float epsilon,
    unsigned int depth,
    unsigned int WH,
    unsigned int quotient, // WH/blockSize
    unsigned int remain, // WH%WARP_SIZE
    unsigned int j,
    unsigned int r1,
    unsigned int dec,
    unsigned int n)
{
  unsigned int tid = threadIdx.x;
  unsigned int channel = blockIdx.x%depth;
  unsigned int blockSize = blockDim.x;

  // Not optimized for compute capability < 1.2
  float mu = g_imean[channel];
  float sigma = g_isigma[channel];
  float g = g_ig[channel];
  float b = g_ib[channel];
  float G = g*rsqrt(sigma+epsilon);

  unsigned int y = blockIdx.x*(WH);
  unsigned int r = y - (y>>MSB_WARP)*WARP_SIZE;//y%WARP_SIZE
  unsigned int i = y + tid -r;// 2* can be added to blockSize, will be modified later
  unsigned int iter = 0;

  if (WH >= blockSize) {
    if (remain == 0) {
      r = WH-quotient*blockSize;//WH%blockSize
      // if r==0 all memory access are coalesced
      if( r==0){
        while (y < n){
          // distinguish the case where WH*K is divisible by blockSize or not
          // WARNING the last warp may have divergent kernels here
          if (iter == quotient-1) {
            g_odata[i] = G*(g_idata[i]-mu) + b;
            y += j;
            i = y +tid;
            iter = 0;
          } else {
            g_odata[i] = G*(g_idata[i]-mu) + b;
            i += blockSize;
            iter++;
          }
        }
      } else {
        while (y < n){
          if (iter == quotient) {
            if(tid<r) g_odata[i] = G*(g_idata[i]-mu) + b;
            y += j;
            i = y+tid;
            iter = 0;
          } else {
            g_odata[i] = G*(g_idata[i]-mu) + b;
            i += blockSize;
            iter++;
          }
        }
      }
    } else {
      // Memory access are made coalesced this imply that some kernel will be waiting while some will be working
      while (y < n){
        // Make all memory access coalesced
        if (iter ==0) {
          //!! WARNING The first warp may have divergent kernels here
          if (tid >= r) {  g_odata[i] = G*(g_idata[i]-mu) + b;}
          else if(r >= dec && tid < r-dec ) { g_odata[i+WH+dec] = G*(g_idata[i+WH+dec]-mu) + b;}
          i += blockSize;
          iter++;
        }
        else if (iter == quotient) {
          if(tid < blockSize - dec + r) g_odata[i] = G*(g_idata[i]-mu) + b;
          r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
          y += j;
          i = y+tid-r;
          iter =0;
        } else{
          g_odata[i] = G*(g_idata[i]-mu) + b;
          i += blockSize;
          iter++;
        }
      }
    }
  } else { // If WH is too small. Highly suboptimal
    if (remain == 0) {
      while (y < n){
        if(tid<WH) {
          g_odata[i] = G*(g_idata[i]-mu) + b;
        }
        y += j;
        i = y+tid;
      }
    } else {
      while (y < n){
        if(tid>=r && tid <  r +WH){ g_odata[i] = G*(g_idata[i]-mu) + b; }
        else if(r+WH > blockSize && tid < WH +r -blockSize) { g_odata[i+blockSize] = G*(g_idata[i+blockSize]-mu) + b; }
        r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
        y += j;
        i = y+tid-r;
      }
    }
  }
}

  template<> vl::Error
vl::impl::bnorm_forward<vl::GPU, float>(Context& context,
    float* output,
    float const* data,
    float* filters,
    float* biaises,
    size_t height, size_t width, size_t depth, size_t size, float epsilon)
{
  unsigned int featureWidth = height*width;
  unsigned int num_threads = VL_CUDA_NUM_THREADS>>1, num_threads_sum = VL_CUDA_NUM_THREADS>>1;

  if(featureWidth < num_threads){
    if(featureWidth>>(MSB_WARP) <4){
      num_threads = 2*WARP_SIZE;
    }
    else if (4<=(featureWidth>>(MSB_WARP)) && (featureWidth>>(MSB_WARP))<8){
      num_threads = 4*WARP_SIZE;
    } else {
      num_threads = 8*WARP_SIZE;
    }
  }

  unsigned int k = featureWidth / num_threads;
  unsigned int remain = featureWidth-(featureWidth>>MSB_WARP)*WARP_SIZE;//featureWidth%WARP_SIZE
  unsigned int mass = featureWidth*size;
  unsigned int volume = mass*depth;
  unsigned int gridDimension = depth*size;
  unsigned int row = size;
  unsigned int dec = (k+1)*num_threads-featureWidth;//num_threads-(featureWidth%num_threads)

  // Compute gridDimension avoid to set too much work groups
  if(gridDimension>65536) {
    if(depth>=65536){
      row = 1;
      gridDimension = depth;
    } else {
      row = 65536/depth+1;
      gridDimension =row*depth;
    }
  }

  unsigned int update = gridDimension*featureWidth;
  unsigned int r1 = update-(update>>MSB_WARP)*WARP_SIZE;

  float *mean, *sigma;
  cudaError_t status;
  vl::Device type = GPU;

  if(gridDimension != depth){
    // intermediate outputs
    unsigned int fin1 = (gridDimension%WARP_SIZE==0) ? gridDimension : WARP_SIZE*((gridDimension>>MSB_WARP)+1);
    float * intermediateOutput = (float*) context.getWorkspace(type, (gridDimension+fin1+2*depth) * sizeof(float)) ;
    mean = intermediateOutput + gridDimension+fin1;
    sigma = mean + depth;

    // Compute mean and variance at the same time
    computeMuSigma<<<gridDimension, num_threads, 2*num_threads*sizeof(float)>>>
      (data, intermediateOutput, depth, row, featureWidth, k, remain, update, r1, dec, volume);

    status = cudaPeekAtLastError() ;
    if(status!= cudaSuccess) mexErrMsgTxt("Error in computeMuSigma") ;

    // Mean and variance computation
    if(row < num_threads_sum){
      if(row>>(MSB_WARP) <4){
        num_threads_sum = 2*WARP_SIZE;
      }
      else if (4<=(row>>(MSB_WARP)) && (row>>(MSB_WARP))<8){
        num_threads_sum = 4*WARP_SIZE;
      } else {
        num_threads_sum = 8*WARP_SIZE;
      }
    }

    computeMatrixSum<<<2*depth,num_threads_sum,num_threads_sum*sizeof(float)>>>
      (intermediateOutput, mean, row, row/num_threads_sum, row%WARP_SIZE, num_threads_sum-(row%num_threads_sum),2*gridDimension);

  } else{
    mean = (float*) context.getWorkspace(type, 2*depth * sizeof(float)) ;
    sigma = mean + depth;

    computeMuSigma<<<gridDimension, num_threads, 2*num_threads*sizeof(float)>>>
      (data, mean, depth, row, featureWidth, k, remain, update, r1, dec, volume);

    status = cudaPeekAtLastError() ;
    if(status!= cudaSuccess) mexErrMsgTxt("Error in computeMuSigma") ;
  }

  divide<<<divideUpwards(depth,num_threads),num_threads>>>
    (mean, mean+depth, (float)mass, depth);

  // Compute output
  normalize<<< gridDimension, num_threads>>>
    (data, mean, sigma, filters, biaises, output, epsilon, depth, featureWidth, k, remain, update, r1, dec, volume);
  status = cudaPeekAtLastError() ;
  if(status!= cudaSuccess) mexErrMsgTxt("Error in normalize") ;

  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ---------------------------------------------------------------- */
/*                                             bnorm_backward       */
/* ---------------------------------------------------------------- */

__global__ void computeMatrixDerSum(
    float * g_idata,
    float * g_omean,
    float * g_oderFilters,
    float * g_oderBiaises,
    unsigned int depth,
    unsigned int row,//row
    unsigned int quotient, // row/blockSize
    unsigned int remain, // row%WARP_SIZE
    unsigned int dec,
    unsigned int n)
{
  extern __shared__ float s[];
  float *mdata = s;
  unsigned int tid = threadIdx.x;
  unsigned int blockidx = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  mdata[tid] = 0;

  accessDataSum(g_idata, mdata, row, quotient, remain, dec, n, blockSize, blockidx, tid);

  if (tid < 32) warpReduce(mdata, tid, blockSize);
  if (tid == 0){
    if(blockidx/(depth>>2)<1){ g_oderFilters[blockidx%(depth>>2)] = mdata[0]; }
    else if ( blockidx/(depth>>2)== 1) { g_oderBiaises[blockidx%(depth>>2)] = mdata[0]; }
    else { g_omean[blockidx%(depth>>1)] = mdata[0]; }
  }
}


__global__ void divideSigma(
    float * dzdg,
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

//template <unsigned int blockSize>
__global__ void computeDerSum(
    float const * g_idata,
    float const * g_iderOutput,
    float * g_imu,
    float * g_odata1,
    float * g_odata2,
    unsigned int depth,
    unsigned int row,
    unsigned int WH,//width*height
    unsigned int quotient, // WH/blockSize
    unsigned int remain, // WH%WARP_SIZE
    unsigned int j,
    unsigned int r1,
    unsigned int dec,
    unsigned int n)
{
  extern __shared__ float s[];
  float *mdata = s;
  unsigned int tid = threadIdx.x;
  unsigned int blockidx = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  float *sdata = (float*)&mdata[blockSize];
  float *rdata = (float*)&sdata[blockSize];
  float *tdata = (float*)&rdata[blockSize];


  unsigned int y = blockidx*(WH);
  unsigned int r = y - (y>>MSB_WARP)*WARP_SIZE;//y%WARP_SIZE
  unsigned int i = y + tid -r;// 2* can be added to blockSize, will be modified later
  unsigned int iter = 0;
  sdata[tid] = 0;
  mdata[tid] = 0;
  rdata[tid] = 0;
  tdata[tid] = 0;

  if (WH >= blockSize) {
    if (remain == 0) {
      r = WH-blockSize*quotient;//WH%blockSize
      // if r==0 all memory access are coalesced
      if( r==0){
        while (y < n){
          if (iter == quotient-1) {
            mdata[tid] += g_idata[i]*g_iderOutput[i];
            sdata[tid] += g_iderOutput[i];
            rdata[tid] += g_idata[i]*g_idata[i];
            tdata[tid] += g_idata[i];
            y += j;
            i = y +tid;
            iter = 0;
          } else {
            mdata[tid] += g_idata[i]*g_iderOutput[i];
            sdata[tid] += g_iderOutput[i];
            rdata[tid] += g_idata[i]*g_idata[i];
            tdata[tid] += g_idata[i];
            i += blockSize;
            iter++;
          }
        }
      } else {
        while (y < n){
          if (iter == quotient) {
            // The last warps may have divergent kernels
            if(tid<r) {
              mdata[tid] += g_idata[i]*g_iderOutput[i];
              sdata[tid] += g_iderOutput[i];
              rdata[tid] += g_idata[i]*g_idata[i];
              tdata[tid] += g_idata[i];
            }
            y += j;
            i = y+tid;
            iter = 0;
          } else {
            mdata[tid] += g_idata[i]*g_iderOutput[i];
            sdata[tid] += g_iderOutput[i];
            rdata[tid] += g_idata[i]*g_idata[i];
            tdata[tid] += g_idata[i];
            i += blockSize;
            iter++;
          }
        }
      }
    } else {
      // Memory access are made coalesced this imply that some kernel will be waiting while some will be working
      while (y < n){
        // Make all memory access coalesced
        if (iter ==0) {
          if (tid >= r){
            mdata[tid] += g_idata[i]*g_iderOutput[i];
            sdata[tid] += g_iderOutput[i];
            rdata[tid] += g_idata[i]*g_idata[i];
            tdata[tid] += g_idata[i];
          }
          else if(r > dec && tid < r-dec ) {
            mdata[tid] += g_idata[i+WH+dec]*g_iderOutput[i+WH+dec];
            sdata[tid] += g_iderOutput[i+WH+dec];
            rdata[tid] += g_idata[i+WH+dec]*g_idata[i+WH+dec];
            tdata[tid] += g_idata[i+WH+dec];
          }
          i += blockSize;
          iter++;
        }
        // if blockSize is too small then can be quotient+1 instead of quotient (arrive when r > blockSize/2 so never in practice)
        else if (iter == quotient) {
          if(tid < blockSize - dec + r){
            mdata[tid] += g_idata[i]*g_iderOutput[i];
            sdata[tid] += g_iderOutput[i];
            rdata[tid] += g_idata[i]*g_idata[i];
            tdata[tid] += g_idata[i];
          }
          r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
          y += j;
          i = y+tid-r;
          iter =0;
        } else{
          mdata[tid] += g_idata[i]*g_iderOutput[i];
          sdata[tid] += g_iderOutput[i];
          rdata[tid] += g_idata[i]*g_idata[i];
          tdata[tid] += g_idata[i];
          i += blockSize;
          iter++;
        }
      }
    }
  } else {
    // If WH is too small. Highly suboptimal
    if (remain == 0) {
      while (y < n){
        // The last warps may have divergent kernels
        if(tid<WH) {
          mdata[tid] += g_idata[i]*g_iderOutput[i];
          sdata[tid] += g_iderOutput[i];
          rdata[tid] += g_idata[i]*g_idata[i];
          tdata[tid] += g_idata[i];
        }
        y += j;
        i = y+tid;
      }
    } else {
      while (y < n){
        if(tid>=r && tid < WH + r){
          mdata[tid] += g_idata[i]*g_iderOutput[i];
          sdata[tid] += g_iderOutput[i];
          rdata[tid] += g_idata[i]*g_idata[i];
          tdata[tid] += g_idata[i];
        }
        else if(WH +r > blockSize && tid < WH +r -blockSize) {
          mdata[tid] += g_idata[i+blockSize]*g_iderOutput[i+blockSize];
          sdata[tid] += g_iderOutput[i+blockSize];
          rdata[tid] += g_idata[i+blockSize]*g_idata[i+blockSize];
          tdata[tid] += g_idata[i+blockSize];
        }
        r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
        y += j;
        i = y+tid-r;
        iter =0;
      }
    }
  }

  // Nothing to optimize here
  __syncthreads();

  if (blockSize >= 1024 && WH + WARP_SIZE >= 512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; rdata[tid] += rdata[tid + 512]; tdata[tid] += tdata[tid + 512];} __syncthreads(); }
  if (blockSize >= 512 && WH + WARP_SIZE >= 256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; rdata[tid] += rdata[tid + 256]; tdata[tid] += tdata[tid + 256];} __syncthreads(); }
  if (blockSize >= 256 && WH + WARP_SIZE >= 128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; rdata[tid] += rdata[tid + 128]; tdata[tid] += tdata[tid + 128];} __syncthreads(); }
  if (blockSize >= 128 && WH + WARP_SIZE >= 64) { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  rdata[tid] += rdata[tid + 64]; tdata[tid] += tdata[tid + 64];} __syncthreads(); }



  if (tid < 32) warp4Reduce(sdata, mdata, rdata, tdata, tid, blockSize);
  if (tid == 0) {
    if(depth==gridDim.x){
      g_odata1[blockidx] = mdata[0];
      g_odata2[blockidx] = sdata[0];
      g_imu[blockidx] = tdata[0];
      g_imu[blockidx+depth] = rdata[0];
    } else{
      i = (blockIdx.x/depth)+(blockIdx.x%depth)*row;
      g_odata1[i] = mdata[0];
      g_odata1[i + gridDim.x] = sdata[0];
      g_odata1[i + 2*gridDim.x] = tdata[0];
      g_odata1[i + 3*gridDim.x] = rdata[0];
    }
  }
}

__global__ void normalizeBackward(
    float const * g_idata,
    float * g_iderOutput,
    float * g_imean,
    float * g_isigma,
    float const * g_ig,
    float * g_imuz,
    float * g_idzdg,
    float * g_odata,
    float epsilon,
    float mass,
    unsigned int depth,
    unsigned int WH,
    unsigned int quotient, // WH/blockSize
    unsigned int remain, // WH%WARP_SIZE
    unsigned int j,
    unsigned int r1,
    unsigned int dec,
    unsigned int n)
{
  unsigned int tid = threadIdx.x;
  unsigned int channel = blockIdx.x%depth;
  unsigned int blockSize = blockDim.x;

  // Not optimized for compute capability < 1.2
  float mu = g_imean[channel];
  float sigma2 = g_isigma[channel] +epsilon;
  float g = g_ig[channel];
  float muz = g_imuz[channel]/mass;
  float dzdg = g_idzdg[channel];
  float G1 = g*rsqrt(sigma2);
  float G2 = (g *dzdg)/(sigma2*mass);

  unsigned int y = blockIdx.x*(WH);
  unsigned int r = y - (y>>MSB_WARP)*WARP_SIZE;//y%WARP_SIZE
  unsigned int i = y + tid -r;// 2* can be added to blockSize, will be modified later
  unsigned int iter = 0;

  if (WH >= blockSize) {
    if (remain == 0) {
      r = WH-quotient*blockSize;//WH%blockSize
      // if r==0 all memory access are coalesced
      if( r==0){
        while (y < n){
          if (iter == quotient-1) {
            g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);
            y += j;
            i = y +tid;
            iter = 0;
          } else {
            g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);
            i += blockSize;
            iter++;
          }
        }
      } else {
        while (y < n){
          if (iter == quotient) {
            if(tid<r) g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);
            y += j;
            i = y+tid;
            iter = 0;
          } else {
            g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);
            i += blockSize;
            iter++;
          }
        }
      }
    } else {
      // Memory access are made coalesced this imply that some kernel will be waiting while some will be working
      while (y < n){
        // Make all memory access coalesced
        if (iter ==0) {
          //!! WARNING The first warp may have divergent kernels here
          if (tid >= r){  g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);}
          else if(r>=dec && tid < r-dec ){ g_odata[i+WH+dec] = G1*(g_iderOutput[i+WH+dec]-muz) - G2*(g_idata[i+WH+dec]-mu); }
          i += blockSize;
          iter++;
        }
        else if (iter == quotient) {
          if(tid < blockSize - dec + r) g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);
          r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
          y += j;
          i = y+tid-r;
          iter =0;
        } else{
          g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);
          i += blockSize;
          iter++;
        }
      }
    }
  } else { // If WH is too small. Highly suboptimal
    if (remain == 0) {
      while (y < n){
        if(tid<WH)  g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);
        y += j;
        i = y+tid;
      }
    } else {
      while (y < n){
        if(tid>=r && tid < WH + r){ g_odata[i] = G1*(g_iderOutput[i]-muz) - G2*(g_idata[i]-mu);}
        else if(r+WH > blockSize && tid < WH +r -blockSize) { g_odata[i+blockSize] = G1*(g_iderOutput[i+blockSize]-muz) - G2*(g_idata[i+blockSize]-mu); }
        r = ((r+r1)>WARP_SIZE) ? (r+r1)-WARP_SIZE : r+r1;
        y += j;
        i = y+tid-r;
        iter =0;
      }
    }
  }
}

  template<> vl::Error
vl::impl::bnorm_backward<vl::GPU, float>(Context& context, float* derData,
    float* derFilters,
    float* derBiaises,
    float const* data,
    float const* filters,
    float const* biaises,
    size_t height, size_t width, size_t depth, size_t size,
    float* derOutput, float epsilon)
{
  unsigned int featureWidth = height*width;
  unsigned int num_threads = VL_CUDA_NUM_THREADS>>1;
  unsigned int num_threads_sum = VL_CUDA_NUM_THREADS>>1;

  if(featureWidth < num_threads){
    if(featureWidth>>(MSB_WARP) <4){
      num_threads = 2*WARP_SIZE;
    }
    else if (4<=(featureWidth>>(MSB_WARP)) && (featureWidth>>(MSB_WARP))<8){
      num_threads = 4*WARP_SIZE;
    } else {
      num_threads = 8*WARP_SIZE;
    }
  }

  unsigned int k = featureWidth / num_threads;
  unsigned int remain = featureWidth-(featureWidth>>MSB_WARP)*WARP_SIZE;//featureWidth%WARP_SIZE
  unsigned int mass = featureWidth*size;
  unsigned int volume = mass*depth;
  unsigned int gridDimension = depth*size;
  unsigned int row = size;
  unsigned int dec = (k+1)*num_threads-featureWidth;//num_threads-(featureWidth%num_threads)

  // Compute gridDimension avoid to set too much work groups
  if(gridDimension>65536) {
    if(depth>=65536){
      // Unlikely to happen
      gridDimension = depth;
    } else {
      row = 65536/depth+1;
      gridDimension =row*depth;
    }
  }

  unsigned int update = gridDimension*featureWidth;
  unsigned int r1 = update-(update>>MSB_WARP)*WARP_SIZE;//update%WARP_SIZE

  vl::Device type = GPU;
  float *intermediateOutput;
  float *mean, *sigma;
  cudaError_t status;

  if(gridDimension != depth){
    // intermediate outputs
    unsigned int fin1 = (gridDimension%WARP_SIZE==0) ? gridDimension : WARP_SIZE*((gridDimension>>MSB_WARP)+1);
    // Might be optimize here to get coalescent access
    intermediateOutput = (float*) context.getWorkspace(type, (3*gridDimension+fin1+2*depth) * sizeof(float)) ;
    mean = intermediateOutput + fin1 + 3*gridDimension;
    sigma = mean + depth;

    status = cudaPeekAtLastError() ;
    if(status!= cudaSuccess) mexErrMsgTxt("Error in computeMuSigma") ;

    // Mean, variance, derFilters and derBiaises computation
    computeDerSum<<<gridDimension, num_threads, 4*num_threads*sizeof(float)>>>
      (data, derOutput, NULL, intermediateOutput, NULL, depth, row, featureWidth, k, remain, update, r1, dec, volume);


    if(row < num_threads_sum){
      if(row>>(MSB_WARP) <4){
        num_threads_sum = 2*WARP_SIZE;
      }
      else if (4<=(row>>(MSB_WARP)) && (row>>(MSB_WARP))<8){
        num_threads_sum = 4*WARP_SIZE;
      } else {
        num_threads_sum = 8*WARP_SIZE;
      }
    }

    // Mean and variance computation
    computeMatrixDerSum<<<4*depth,num_threads_sum,num_threads_sum*sizeof(float)>>>
      (intermediateOutput, mean, derFilters, derBiaises, 4*depth, row, row/num_threads_sum, row%WARP_SIZE, num_threads_sum-(row%num_threads_sum),4*gridDimension);

    status = cudaPeekAtLastError() ;
    if(status!= cudaSuccess) mexErrMsgTxt("Error in computeMuSigma") ;

  } else{
    mean = (float*) context.getWorkspace(type, (2*depth) * sizeof(float)) ;
    sigma = mean + depth;

    computeDerSum<<<gridDimension, num_threads, 4*num_threads*sizeof(float)>>>
      (data, derOutput, mean, derFilters, derBiaises, depth, row, featureWidth, k, remain, update, r1, dec, volume);

    status = cudaPeekAtLastError() ;
    if(status!= cudaSuccess) mexErrMsgTxt("Error in computeDerSum") ;
  }

  divideSigma<<<divideUpwards(depth,num_threads),num_threads>>>(derFilters, derBiaises, mean,sigma,epsilon,(float)mass,depth);

  // Compute output
  normalizeBackward<<< gridDimension, num_threads>>>
    (data, derOutput, mean, sigma, filters, derBiaises, derFilters, derData,
     epsilon, (float)mass, depth, featureWidth, k, remain, update, r1, dec, volume);

  status = cudaPeekAtLastError() ;
  if(status!= cudaSuccess) mexErrMsgTxt("Error in normalizeBackward") ;

  // Delete dynamic variables
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
