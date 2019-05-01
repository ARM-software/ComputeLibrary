/*
 * Copyright (c) 2016-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __ARM_COMPUTE_NELAPLACIANPYRAMID_H__
#define __ARM_COMPUTE_NELAPLACIANPYRAMID_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConvertLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGaussian5x5.h"
#include "arm_compute/runtime/NEON/functions/NEGaussianPyramid.h"
#include "arm_compute/runtime/Pyramid.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to execute laplacian pyramid. This function calls the following NEON kernels and functions:
 *
 * -# @ref NEGaussianPyramidHalf
 * -# @ref NEGaussian5x5
 * -# @ref NEArithmeticSubtraction
 *
 *  First a Gaussian pyramid is created. Then, for each level i, the corresponding tensor I(i) is blurred with the Gaussian 5x5 filter, and then
 *  difference between the two tensors is the corresponding level L(i) of the Laplacian pyramid.
 *  L(i) = I(i) - Gaussian5x5(I(i))
 *  Level 0 has always the same first two dimensions as the input tensor.
*/
class NELaplacianPyramid : public IFunction
{
public:
    /** Constructor */
    NELaplacianPyramid();
    /** Initialise the function's source, destinations and border mode.
     *
     * @param[in]  input                 Source tensor. Data type supported: U8.
     * @param[out] pyramid               Destination pyramid tensors, Data type supported at each level: S16.
     * @param[out] output                The lowest resolution tensor necessary to reconstruct the input tensor from the pyramid. Data type supported: S16.
     *                                   The first two dimensions of this tensor must match the first two dimensions of the tensor in the last level of the pyramid, that is:
     *                                   out.width = in.width() / pow(2,pyramid_levels-1) and out.height = in.height() / pow(2,pyramid_levels-1)
     * @param[in]  border_mode           Border mode to use.
     * @param[in]  constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(const ITensor *input, IPyramid *pyramid, ITensor *output, BorderMode border_mode, uint8_t constant_border_value);

    // Inherited methods overridden:
    void run() override;

private:
    size_t                               _num_levels;
    NEGaussianPyramidHalf                _gaussian_pyr_function;
    std::vector<NEGaussian5x5>           _convf;
    std::vector<NEArithmeticSubtraction> _subf;
    Pyramid                              _gauss_pyr;
    Pyramid                              _conv_pyr;
    NEDepthConvertLayer                  _depth_function;
};
}
#endif /*__ARM_COMPUTE_NELAPLACIANPYRAMID_H__ */
