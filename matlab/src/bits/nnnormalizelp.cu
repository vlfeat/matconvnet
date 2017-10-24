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
  std::vector<Int> steps ;
  std::vector<Int> stepPeriods ;
  Int normsVolume ;
  Int inputVolume ;
} ;

VisitPattern getVisitPatternForInput(NormalizeLp const & op, vl::Tensor input)
{
  // Compute tensor geometry.
  auto n = (size_t)input.getNumDimensions() ;
  auto inputDimensions = std::vector<size_t>(begin(input.getDimensions()),
                                             end(input.getDimensions())) ;

  assert(n <= 4) ; // Todo: relax (just extend the for loops below).

  Int inputVolume = 1 ;
  Int normsVolume = 1 ;
  auto steps = std::vector<Int>(n+1,0) ;
  auto stepPeriods = std::vector<Int>(n+1,0) ;

  // Find out how to traverse the reduced results as the input is
  // scanned from first to last element.
  for (size_t d = 0 ; d < n ; ++d) {
    stepPeriods[size_t(d)] = inputVolume ;

    auto const& sd = op.getSelectedDimensions() ;
    bool squashed = (find(sd.begin(), sd.end(), d) != sd.end()) ;

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
  for (size_t d = 0 ; d + 2 < steps.size() ; ) {
    if (steps[d] == 0 && steps[d+1] == 0) {
      steps.erase(steps.begin() + as_signed(d)) ;
      stepPeriods.erase(stepPeriods.begin() + as_signed(d) + 1) ;
    } else {
      ++ d ;
    }
  }

  // Make it suitable for more efficient loops.
  for (Int d = as_signed(steps.size()) - 1 ; d >= 1 ; --d) {
    stepPeriods[as_unsigned(d)] /= stepPeriods[as_unsigned(d) - 1] ;
  }
  for (Int d = as_signed(steps.size()) ; d < 5 ; ++d) {
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
  memset(normsData, 0, as_unsigned(vp.normsVolume) * sizeof(type)) ;

  // Accumulate norm.
  auto const exponent = static_cast<type>(op.getExponent()) ;
  auto const epsilon = static_cast<type>(op.getEpsilon()) ;
  auto npt = normsData ;
  auto ipt = inputData ;
  for (Int i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
    for (Int i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
      for (Int i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
        for (Int i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
          *npt += pow(*ipt++, exponent) ;
          npt += vp.steps[0] ;
        }
        npt += vp.steps[1] ;
      }
      npt += vp.steps[2] ;
    }
    npt += vp.steps[3] ;
  }

  // Root norm.
  for (Int i = 0 ; i < vp.normsVolume ; ++i) {
    normsData[i] = std::pow(normsData[i] + (type)epsilon, static_cast<type>(1.0/exponent)) ;
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
    else { normsData = new type [as_unsigned(vp.normsVolume)] ; normsDataIsOwner = true ; }

    // Compute norm if needed.
    if (!givenNorms) {
      computeNorms(op,normsData,inputData,vp) ;
    }

    // Divide norm.
    if (output) {
      auto npt = normsData ;
      type * outputData = (type*)output.getMemory() ;
      for (Int i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
        for (Int i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
          for (Int i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
            for (Int i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
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
    else { normsData = new type [as_unsigned(vp.normsVolume)] ; normsDataIsOwner = true ; }

    // Compute norms if given.
    if (!givenNorms) {
      computeNorms(op,normsData,inputData,vp) ;
    }

    // Compute sum(derOutput .* input).
    type * scratchData = new type [as_unsigned(vp.normsVolume)] () ; // zeros
    {
      auto ipt = inputData ;
      auto dopt = derOutputData ;
      auto spt = scratchData ;
      for (Int i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
        for (Int i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
          for (Int i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
            for (Int i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
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
      auto const exponent = static_cast<type>(op.getExponent()) ;
      auto dipt = derInputData ;
      auto npt = normsData ;
      auto ipt = inputData ;
      auto dopt = derOutputData ;
      auto spt = scratchData ;
      for (Int i3 = 0 ; i3 < vp.stepPeriods[4] ; ++i3) {
        for (Int i2 = 0 ; i2 < vp.stepPeriods[3] ; ++i2) {
          for (Int i1 = 0 ; i1 < vp.stepPeriods[2] ; ++i1) {
            for (Int i0 = 0 ; i0 < vp.stepPeriods[1] ; ++i0) {
              auto n = *npt ;
              *dipt = (*dopt) / n - (*spt) * std::pow(*ipt, exponent-1) / std::pow(n,exponent+1) ;

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
                         std::vector<Int> const& selectedDimensions,
                         double exponent,
                         double epsilon) :
Operation(context),
selectedDimensions(selectedDimensions),
exponent(exponent),
epsilon(epsilon)
{ }

vl::TensorShape
NormalizeLp::getNormsShapeForData(vl::Tensor const &data)
{
  vl::TensorShape shape(data) ;
  auto n = shape.getNumDimensions() ;
  for (size_t d = 0 ; d < size_t(n) ; ++d) {
    bool squashed =
    (find(selectedDimensions.begin(), selectedDimensions.end(), d) !=
     selectedDimensions.end()) ;
    if (squashed) { shape.setDimension((Int)d, 1) ; }
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