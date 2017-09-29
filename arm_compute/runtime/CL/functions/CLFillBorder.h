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
#ifndef __ARM_COMPUTE_CLFILLBORDER_H__
#define __ARM_COMPUTE_CLFILLBORDER_H__

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLFillBorderKernel */
class CLFillBorder : public ICLSimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in,out] tensor                Source tensor. Data types supported: QS8/U8/S16/QS16
     * @param[in]     border_width          The border width
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *tensor, unsigned int border_width, BorderMode border_mode, const PixelValue &constant_border_value = PixelValue());
};
}
#endif /*__ARM_COMPUTE_FILLBORDER_H__ */
