/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEBOX3x3_H__
#define __ARM_COMPUTE_NEBOX3x3_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to execute box filter 3x3. This function calls the following NEON kernels:
 *
 *  -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 *  -# @ref NEBox3x3Kernel
 *
 */
class NEBox3x3 : public INESimpleFunction
{
public:
    /** Initialise the function's input, output and border mode.
     *
     * @note The border handler is run on the input tensor.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Destination tensor, Data type supported: U8.
     * @param[in]      border_mode           Strategy to use for borders.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     * @param[in]      use_fp16              (Optional) If true the FP16 kernels will be used. If false F32 kernels are used.
     */
    void configure(ITensor *input, ITensor *output, BorderMode border_mode, uint8_t constant_border_value = 0, bool use_fp16 = false);
};
}
#endif /*__ARM_COMPUTE_NEBOX3x3_H__ */
