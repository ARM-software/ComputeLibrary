/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEFILLBORDER_H__
#define __ARM_COMPUTE_NEFILLBORDER_H__

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEFillBorderKernel */
class NEFillBorder : public IFunction
{
public:
    /** Initialize the function's source, destination and border_mode.
     *
     * @note This function fills the borders within the XY-planes.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8/QS8/S16/S32/F32
     * @param[in]      border_width          Width of the tensor border in pixels.
     * @param[in]      border_mode           Strategy to use for borders.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, unsigned int border_width, BorderMode border_mode, const PixelValue &constant_border_value = PixelValue());

    // Inherited methods overridden:
    void run() override;

private:
    NEFillBorderKernel _border_handler; /**< Kernel to handle image borders */
};
}
#endif /*__ARM_COMPUTE_NEFILLBORDER_H__ */
