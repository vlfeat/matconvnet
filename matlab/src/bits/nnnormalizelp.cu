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

template<bool givenNomrs> struct NormAgrument ;
template<> struct NormAgrument<true> { typedef vl::Tensor const &type ; } ;
template<> struct NormAgrument<false> { typedef vl::Tensor &type ; } ;

#if ENABLE_GPU
#include "nnnormalizelp_gpu.cu"
#endif

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
  auto const & inputDimensions = input.getDimensions() ;

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

template<vl::DataType dataType, bool givenNorms>
struct NormalizeLpForwardCPU
{
  vl::ErrorCode operator()(vl::nn::NormalizeLp const& op,
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
  vl::ErrorCode operator()(vl::nn::NormalizeLp const&op,
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

NormalizeLp::NormalizeLp(vl::Context &context,
                         std::vector<Int> const& selectedDimensions,
                         double exponent,
                         double epsilon) :
Operation(context),
selectedDimensions(selectedDimensions),
exponent(exponent),
epsilon(epsilon)
{ }

NormalizeLp::NormalizeLp(vl::Context& context)
:
Operation(context),
selectedDimensions(1,2),
epsilon(1e-2),
exponent(2.0)
{ }

ErrorCode NormalizeLp::setSelectedDimensions(std::vector<Int> const& sel)
{
  // Selector must be non-negative.
  if (any_of(begin(sel),end(sel),[](Int x){return x < 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "ConvolutionTranspose: An element of SELECTEDIMENSIONS is less than 0.") ;
  }
  selectedDimensions = sel ;
  sort(begin(selectedDimensions),end(selectedDimensions)) ;
  selectedDimensions.erase
  (unique(begin(selectedDimensions),end(selectedDimensions)),
   end(selectedDimensions)) ;
  return VLE_Success ;
}

vl::ErrorCode
NormalizeLp::forwardShape(vl::TensorShape &output,
                          vl::TensorShape &norms,
                          vl::TensorShape const &data) const
{
  output = data ;
  norms = data ;
  vl::TensorShape shape(data) ;
  auto n = norms.getNumDimensions() ;
  for (size_t d = 0 ; d < size_t(n) ; ++d) {
    bool squashed =
    (find(selectedDimensions.begin(), selectedDimensions.end(), d) !=
     selectedDimensions.end()) ;
    if (squashed) { norms.setDimension((Int)d, 1) ; }
  }
  return vl::VLE_Success ;
}

template <bool optionalNorms>
static vl::ErrorCode
check_helper(NormalizeLp const& op,
             vl::Tensor const &output,
             vl::Tensor const &norms,
             vl::Tensor const &input)
{
  // Check the tensor consistency.
  if (!check_tensor_compatibility(output,norms,input)) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "The tensors have mismatching data or device type.") ;
  }

  // Check the data.
  if (output.isEmpty() | output.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "OUTPUT or DEROUTPUT is empty or null.") ;
  }
  if (optionalNorms) {
    if (!norms.isEmpty() && norms.isNull()) {
      return op.getContext().setError
      (VLE_IllegalArgument,
       "NORMS is non empty but null.") ;
    }
  } else {
    if (norms.isEmpty() || norms.isNull()) {
      return op.getContext().setError
      (VLE_IllegalArgument,
       "NORMS is empty or null.") ;
    }
  }
  if (input.isEmpty() | input.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "INPUT is emtpy or null.") ;
  }

  // Check the tensor shape.
  vl::ErrorCode error ;
  TensorShape outputShape ;
  TensorShape normShape ;
  if ((error = op.forwardShape(outputShape, normShape, input.getShape())) != VLE_Success) {
    return error ;
  }
  if (output.getShape() != outputShape) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "OUTPUT or DEROUTPUT do not have the appropriate dimensions.") ;
  }
  if (!norms.isEmpty() && (norms.getNumElements() != normShape.getNumElements())) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "NORMS does not have the appropriate dimensions.") ;
  }
  return VLE_Success ;
}

template <bool optionalNorms>
static vl::ErrorCode
check_helper_backward(NormalizeLp const& op,
                      Tensor const &derInput,
                      Tensor const &norm,
                      Tensor const &input,
                      Tensor const &derOutput)
{
  vl::ErrorCode error = check_helper<optionalNorms>(op,derOutput,norm,input) ;
  if (error != vl::VLE_Success) {
    return error ;
  }
  // Check the tensor consistency.
  if (!check_tensor_compatibility(derInput,norm,input,derOutput)) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "The tensors have mismatching data or device type.") ;
  }

  // Check the data.
  if (derInput.isEmpty() | derInput.isNull()) {
    return op.getContext().setError
    (VLE_IllegalArgument,
     "DERINPUT is empty or null.") ;
  }

  // Check the tensor shape.
  if (derInput.getShape() != input.getShape()) {
    return op.getContext().setError
    (VLE_TensorShapeMismatch,
     "DERINPUT does not have the appropriate dimensions.") ;
  }
  return VLE_Success ;
}

vl::ErrorCode
NormalizeLp::forward(vl::Tensor &output,
                     vl::Tensor &norms,
                     vl::Tensor const &data) const
{
  vl::ErrorCode error = check_helper<true>(*this,output,norms,data) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"NormalizeLpForward") ;
  }

  VLLOG(*this,1)
  << "NormalizeLpForward:"
  << " selected dims=" << pretty(getSelectedDimensions())
  << " exponent=" << getExponent()
  << " epsilon=" << getEpsilon() ;

  VLLOG(*this,1)
  << "NormalizeLpForward:"
  << " data=" << pretty(data.getDimensions())
  << " norms=" << pretty(norms.getDimensions())
  << " output=" << pretty(output.getDimensions()) ;

  return getContext().passError
  (dispatch<NormalizeLpForward>()(*this,output,norms,data),
   "NormalizeLpForward") ;
}

vl::ErrorCode
NormalizeLp::forwardWithNorms(vl::Tensor &output,
                              vl::Tensor const &norms,
                              vl::Tensor const &data) const
{
  vl::ErrorCode error = check_helper<false>(*this,output,norms,data) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"NormalizeLpForwardWithNorms") ;
  }

  VLLOG(*this,1)
  << "NormalizeLpForwardWithNorms:"
  << " selected dims=" << pretty(getSelectedDimensions())
  << " exponent=" << getExponent()
  << " epsilon=" << getEpsilon() ;

  VLLOG(*this,1)
  << "NormalizeLpForwardWithNorms:"
  << " data=" << pretty(data.getDimensions())
  << " norms=" << pretty(norms.getDimensions())
  << " output=" << pretty(output.getDimensions()) ;

  return getContext().passError
  (dispatch<NormalizeLpForwardWithNorms>()(*this,output,norms,data),
   "NormalizeLpForwardWithNorms") ;
}

vl::ErrorCode
NormalizeLp::backward(vl::Tensor &derData,
                      vl::Tensor &norms,
                      vl::Tensor const &data,
                      vl::Tensor const &derOutput) const
{
  vl::ErrorCode error = check_helper_backward<true>(*this,derData,norms,data,derOutput) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"NormalizeLpBackward") ;
  }

  VLLOG(*this,1)
  << "NormalizeLpBackward:"
  << " selected dims=" << pretty(getSelectedDimensions())
  << " exponent=" << getExponent()
  << " epsilon=" << getEpsilon() ;

  VLLOG(*this,1)
  << "NormalizeLpBackward:"
  << " derData=" << pretty(derData.getDimensions())
  << " norms=" << pretty(norms.getDimensions())
  << " data=" << pretty(data.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  return getContext().passError
  (dispatch<NormalizeLpBackward>()(*this,derData,norms,data,derOutput),
   "NormalizeLpBackward");
}

vl::ErrorCode
NormalizeLp::backwardWithNorms(vl::Tensor &derData,
                               vl::Tensor const &norms,
                               vl::Tensor const &data,
                               vl::Tensor const &derOutput) const
{
  vl::ErrorCode error = check_helper_backward<false>(*this,derData,norms,data,derOutput) ;
  if (error != VLE_Success) {
    return getContext().passError(error,"NormalizeLpBackwardWithNorms") ;
  }

  VLLOG(*this,1)
  << "NormalizeLpBackwardWithNorms:"
  << " selected dims=" << pretty(getSelectedDimensions())
  << " exponent=" << getExponent()
  << " epsilon=" << getEpsilon() ;

  VLLOG(*this,1)
  << "NormalizeLpBackwardWithNorms:"
  << " derData=" << pretty(derData.getDimensions())
  << " norms=" << pretty(norms.getDimensions())
  << " data=" << pretty(data.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  return getContext().passError
  (dispatch<NormalizeLpBackwardWithNorms>()(*this,derData,norms,data,derOutput),
   "NormalizeLpBackwardWithNorms") ;
}
