/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NENONLINEARFILTER_H
#define ARM_COMPUTE_NENONLINEARFILTER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to execute non linear filter. This function calls the following Neon kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NENonLinearFilterKernel
 *
 * @note Supported mask dimensions squares of sizes 3, 5
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NENonLinearFilter : public INESimpleFunction
{
public:
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Destination tensor. Data type supported: U8
     * @param[in]      function              Non linear function to perform
     * @param[in]      mask_size             Mask size. Supported sizes: 3, 5
     * @param[in]      pattern               Mask pattern
     * @param[in]      mask                  The given mask. Will be used only if pattern is specified to PATTERN_OTHER
     * @param[in]      border_mode           Strategy to use for borders.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode,
                   uint8_t constant_border_value = 0);
};
}
#endif /*ARM_COMPUTE_NENONLINEARFILTER_H */
