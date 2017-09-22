// @file nnnormalizelp.cu
// @brief Batch normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnnormalizelp.hpp"
#include "impl/dispatcher.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<vl::DeviceType deviceType, vl::DataType dataType> struct NormalizeLpForward ;
template<vl::DeviceType deviceType, vl::DataType dataType> struct NormalizeLpForwardWithNorms ;
template<vl::DeviceType deviceType, vl::DataType dataType> struct NormalizeLpBackward ;
template<vl::DeviceType deviceType, vl::DataType dataType> struct NormalizeLpBackwardWithNorms ;

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

struct VisitPattern
{
  std::vector<ptrdiff_t> steps ;
  std::vector<ptrdiff_t> stepPeriods ;
  size_t normsVolume ;
  size_t inputVolume ;
} ;

VisitPattern getVisitPatternForInput(NormalizeLp const & op, vl::Tensor input)
{
  // Compute tensor geometry.
  int n = input.getNumDimensions() ;
  auto inputDimensions = std::vector<size_t>(input.getDimensions(),
                                             input.getDimensions() + n) ;

  assert(n <= 4) ; // Todo: relax (just extend the for loops below).

  size_t inputVolume = 1 ;
  size_t normsVolume = 1 ;
  auto steps = std::vector<ptrdiff_t>(n+1,0) ;
  auto stepPeriods = std::vector<ptrdiff_t>(n+1,0) ;

  // Find out how to traverse the reduced results as the input is
  // scanned from first to last element.
  for (int d = 0 ; d < n ; ++d) {
    stepPeriods[d] = inputVolume ;

    bool squashed =
    (find(op.selectedDimensions.begin(), op.selectedDimensions.end(), d) !=
     op.selectedDimensions.end()) ;

    if (!squashed)  {
      steps[d] += normsVolume ;
      normsVolume *= inputDimensions[d] ;
      steps[d+1] -= normsVolume ;
    }
    inputVolume *= inputDimensions[d] ;
  }
  steps[n] = 0 ;
  stepPeriods[n] = inputVolume ;

  // Simplify traversal.
  for (int d = 0 ; d < steps.size() - 2 ; ) {
    if (steps[d] == 0 && steps[d+1] == 0) {
      steps.erase(steps.begin() + d) ;
      stepPeriods.erase(stepPeriods.begin() + d+1) ;
    } else {
      ++ d ;
    }
  }

  // Make it suitable for more efficient loops.
  for (int d = steps.size()-1 ; d >= 1 ; --d) {
    stepPeriods[d] /= stepPeriods[d - 1] ;
  }
  for (int d = steps.size() ; d < 5 ; ++d) {
    steps.push_back(0) ;
    stepPeriods.push_back(1) ;
  }

  VisitPattern vp ;
  vp.steps = move(steps) ;
  vp.stepPeriods = move(stepPeriods) ;
  vp.inputVolume = inputVolume ;
  vp.normsVolume = normsVolume ;
  return vp ;
}

template<typename type>
void computeNorms(NormalizeLp const & op,
                  type * normsData, type const * inputData, VisitPattern vp)
{
  // Clear norms.
  memset(normsData, 0, vp.normsVolume * sizeof(type)) ;

  // Accumulate norm.
  auto npt = normsData ;
  auto ipt = inputData ;
  for (ptrdiff_t i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
    for (ptrdiff_t i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
      for (ptrdiff_t i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
        for (ptrdiff_t i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
          *npt += pow(*ipt++, op.exponent) ;
          npt += vp.steps[0] ;
        }
        npt += vp.steps[1] ;
      }
      npt += vp.steps[2] ;
    }
    npt += vp.steps[3] ;
  }

  // Root norm.
  for (ptrdiff_t i = 0 ; i < vp.normsVolume ; ++i) {
    normsData[i] = pow(normsData[i] + op.epsilon, 1.0/op.exponent) ;
  }
}

// -------------------------------------------------------------------
//                                                         CPU forward
// -------------------------------------------------------------------


template<bool givenNomrs> struct NormAgrument ;
template<> struct NormAgrument<true> { typedef vl::Tensor const &type ; } ;
template<> struct NormAgrument<false> { typedef vl::Tensor &type ; } ;

template<vl::DataType dataType, bool givenNorms>
struct NormalizeLpForwardCPU
{
  vl::ErrorCode operator()(vl::nn::NormalizeLp & op,
                           vl::Tensor &output,
                           typename NormAgrument<givenNorms>::type norms,
                           vl::Tensor const &input)
  {
    assert(norms || !givenNorms) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto vp = getVisitPatternForInput(op, input) ;

    type const * inputData = (type const*)input.getMemory() ;
    type * normsData ;
    bool normsDataIsOwner = false ;
    if (norms) { normsData = (type*)norms.getMemory() ; }
    else { normsData = new type [vp.normsVolume] ; normsDataIsOwner = true ; }

    // Compute norm if needed.
    if (!givenNorms) {
      computeNorms(op,normsData,inputData,vp) ;
    }

    // Divide norm.
    if (output) {
      auto npt = normsData ;
      type * outputData = (type*)output.getMemory() ;
      for (ptrdiff_t i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
        for (ptrdiff_t i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
          for (ptrdiff_t i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
            for (ptrdiff_t i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
              *outputData = *inputData / *npt ;
              inputData ++ ;
              outputData ++ ;
              npt += vp.steps[0] ;
            }
            npt += vp.steps[1] ;
          }
          npt += vp.steps[2] ;
        }
        npt += vp.steps[3] ;
      }
    }

    // Finish.
    if (normsData && normsDataIsOwner) {
      delete [] normsData ;
    }
    return vl::VLE_Success ;
  }
} ;

template<vl::DataType dataType>
struct NormalizeLpForward<vl::VLDT_CPU, dataType>
: public NormalizeLpForwardCPU<dataType,false>
{ } ;

template<vl::DataType dataType>
struct NormalizeLpForwardWithNorms<vl::VLDT_CPU, dataType>
: public NormalizeLpForwardCPU<dataType,true>
{ } ;

// -------------------------------------------------------------------
//                                                        CPU backward
// -------------------------------------------------------------------

template<vl::DataType dataType, bool givenNorms>
struct NormalizeLpBackwardCPU
{
  vl::ErrorCode operator()(vl::nn::NormalizeLp &op,
                           vl::Tensor &derInput,
                           typename NormAgrument<givenNorms>::type norms,
                           vl::Tensor const &input,
                           vl::Tensor const& derOutput)
  {
    assert(norms || !givenNorms) ;

    // Compute tensor geometry.
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto vp = getVisitPatternForInput(op, input) ;

    auto derInputData = (type*)derInput.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto derOutputData = (type const *)derOutput.getMemory() ;
    type * normsData ;
    bool normsDataIsOwner = false ;
    if (norms) { normsData = (type*)norms.getMemory() ; }
    else { normsData = new type [vp.normsVolume] ; normsDataIsOwner = true ; }

    // Compute norms if given.
    if (!givenNorms) {
      computeNorms(op,normsData,inputData,vp) ;
    }

    // Compute sum(derOutput .* input).
    type * scratchData = new type [vp.normsVolume] () ; // zeros
    {
      auto ipt = inputData ;
      auto dopt = derOutputData ;
      auto spt = scratchData ;
      for (ptrdiff_t i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
        for (ptrdiff_t i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
          for (ptrdiff_t i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
            for (ptrdiff_t i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
              *spt += (*ipt) * (*dopt) ;
              ipt ++ ;
              dopt ++ ;
              spt += vp.steps[0] ;
            }
            spt += vp.steps[1] ;
          }
          spt += vp.steps[2] ;
        }
        spt += vp.steps[3] ;
      }
    }

    // Compute derInputs.
    {
      auto dipt = derInputData ;
      auto npt = normsData ;
      auto ipt = inputData ;
      auto dopt = derOutputData ;
      auto spt = scratchData ;
      for (ptrdiff_t i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
        for (ptrdiff_t i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
          for (ptrdiff_t i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
            for (ptrdiff_t i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
              auto n = *npt ;
              *dipt = (*dopt) / n - (*spt) * pow(*ipt, op.exponent-1) / pow(n,op.exponent+1) ;

              dipt ++ ;
              ipt ++ ;
              dopt ++ ;

              npt += vp.steps[0] ;
              spt += vp.steps[0] ;
            }
            npt += vp.steps[1] ;
            spt += vp.steps[1] ;
          }
          npt += vp.steps[2] ;
          spt += vp.steps[2] ;
        }
        npt += vp.steps[3] ;
        spt += vp.steps[3] ;
      }
    }

    // Finish.
    if (normsData && normsDataIsOwner) {
      delete [] normsData ;
    }
    delete [] scratchData ;
    return vl::VLE_Success ;
  }
} ;

template<vl::DataType dataType>
struct NormalizeLpBackward<vl::VLDT_CPU, dataType>
: public NormalizeLpBackwardCPU<dataType,false>
{ } ;

template<vl::DataType dataType>
struct NormalizeLpBackwardWithNorms<vl::VLDT_CPU, dataType>
: public NormalizeLpBackwardCPU<dataType,true>
{ } ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_GPU
#include "nnnormalizelp_gpu.cu"
#endif

NormalizeLp::NormalizeLp(vl::Context &context,
                         std::vector<int> const& selectedDimensions,
                         double exponent,
                         double epsilon) :
context(context),
selectedDimensions(selectedDimensions),
exponent(exponent),
epsilon(epsilon)
{ }

vl::TensorShape
NormalizeLp::getNormsShapeForData(vl::Tensor const &data)
{
  vl::TensorShape shape(data) ;
  int n = shape.getNumDimensions() ;
  for (int d = 0 ; d < n ; ++d) {
    bool squashed =
    (find(selectedDimensions.begin(), selectedDimensions.end(), d) !=
     selectedDimensions.end()) ;
    if (squashed) { shape.setDimension(d, 1) ; }
  }
  return shape ;
}

vl::ErrorCode
NormalizeLp::forward(vl::Tensor &output,
                     vl::Tensor &norms,
                     vl::Tensor const &data)
{
  return dispatch<NormalizeLpForward>()(*this,output,norms,data) ;
}

vl::ErrorCode
NormalizeLp::forwardWithNorms(vl::Tensor &output,
                              vl::Tensor const &norms,
                              vl::Tensor const &data)
{
  return dispatch<NormalizeLpForwardWithNorms>()(*this,output,norms,data) ;
}

vl::ErrorCode
NormalizeLp::backward(vl::Tensor &derData,
                      vl::Tensor &norms,
                      vl::Tensor const &data,
                      vl::Tensor const &derOutput)
{
  return dispatch<NormalizeLpBackward>()(*this,derData,norms,data,derOutput) ;
}

vl::ErrorCode
NormalizeLp::backwardWithNorms(vl::Tensor &derData,
                               vl::Tensor const &norms,
                               vl::Tensor const &data,
                               vl::Tensor const &derOutput)
{
  return dispatch<NormalizeLpBackwardWithNorms>()(*this,derData,norms,data,derOutput) ;
}
