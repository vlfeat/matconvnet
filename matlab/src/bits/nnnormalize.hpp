//
//  nnnormalize.hpp
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__nnnormalize__
#define __matconv__nnnormalize__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  int nnnormalize_forward(vl::Context& context,
                          vl::Tensor output,
                          vl::Tensor data,
                          size_t normDetph,
                          double kappa, double alpha, double beta) ;

  int nnnormalize_backward(vl::Context& context,
                           vl::Tensor derData,
                           vl::Tensor data,
                           vl::Tensor derOutput,
                           size_t normDetph,
                           double kappa, double alpha, double beta) ;
}

#endif /* defined(__matconv__nnnormalize__) */
