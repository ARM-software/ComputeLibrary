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
#ifndef ARM_COMPUTE_NEFILLBORDER_H
#define ARM_COMPUTE_NEFILLBORDER_H

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
// Forward declaration
class ITensor;
class NEFillBorderKernel;

/** Basic function to run @ref NEFillBorderKernel */
class NEFillBorder : public IFunction
{
public:
    NEFillBorder();
    /** Initialize the function's source, destination and border_mode.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @note This function fills the borders within the XY-planes.
     *
     * @param[in, out] input                 Source tensor. Data type supported: All
     * @param[in]      border_width          Width of the tensor border in pixels.
     * @param[in]      border_mode           Strategy to use for borders.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, unsigned int border_width, BorderMode border_mode, const PixelValue &constant_border_value = PixelValue());

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEFillBorderKernel> _border_handler; /**< Kernel to handle image borders */
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEFILLBORDER_H */
