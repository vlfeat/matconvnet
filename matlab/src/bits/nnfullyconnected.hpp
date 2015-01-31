//
//  nnconv.h
//  matconv
//
//  Created by Andrea Vedaldi on 04/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__nnfullyconnected__
#define __matconv__nnfullyconnected__

#include "data.hpp"

namespace vl {

  int nnfullyconnected_forward(vl::Context& context,
                               vl::Tensor output,
                               vl::Tensor data,
                               vl::Tensor filters,
                               vl::Tensor biases) ;

  int nnfullyconnected_backward(vl::Context& context,
                                vl::Tensor derData,
                                vl::Tensor derFilters,
                                vl::Tensor derBiases,
                                vl::Tensor data,
                                vl::Tensor filters,
                                vl::Tensor derOutput) ;
}


#endif /* defined(__matconv__nnfullyconnected__) */
