/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NENONMAXIMASUPPRESSION3X3_H
#define ARM_COMPUTE_NENONMAXIMASUPPRESSION3X3_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to execute non-maxima suppression over a 3x3 window. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NENonMaximaSuppression3x3Kernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NENonMaximaSuppression3x3 : public INESimpleFunction
{
public:
    /** Initialise the function's source, destinations and border mode.
     *
     * @note The implementation supports just 2 border modes: UNDEFINED and CONSTANT
     *       The constant values used with CONSTANT border mode is 0
     *
     * @param[in, out] input       Source tensor. Data type supported: U8/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output      Destination for the Non-Maxima suppressions 3x3. Data type supported: same as @p input
     * @param[in]      border_mode Border mode to use for non-maxima suppression. The implementation supports just 2 border modes: UNDEFINED and CONSTANT
     *
     */
    void configure(ITensor *input, ITensor *output, BorderMode border_mode);
};
}
#endif /* ARM_COMPUTE_NENONMAXIMASUPPRESSION3X3_H */
