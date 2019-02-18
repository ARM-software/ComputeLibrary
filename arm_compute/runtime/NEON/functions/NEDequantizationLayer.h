/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDEQUANTIZATIONLAYER_H__
#define __ARM_COMPUTE_NEDEQUANTIZATIONLAYER_H__

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to run @ref NEDequantizationLayerKernel that dequantizes an input tensor */
class NEDequantizationLayer : public INESimpleFunctionNoBorder
{
public:
    /** Configure the kernel.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8.
     * @param[out] output Destination tensor with the same dimensions of input. Data type supported: F16/F32.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDequantizationLayer
     *
     * @param[in] input  Input tensor info. Data types supported: QASYMM8.
     * @param[in] output Output tensor info. Data type supported: F16/F32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDEQUANTIZATIONLAYER_H__ */
