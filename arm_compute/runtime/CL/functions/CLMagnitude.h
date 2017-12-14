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
#ifndef __ARM_COMPUTE_CLMAGNITUDE_H__
#define __ARM_COMPUTE_CLMAGNITUDE_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLMagnitudePhaseKernel. */
class CLMagnitude : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs.
     *
     * @param[in]  input1   First tensor input. Data types supported: S16.
     * @param[in]  input2   Second tensor input. Data types supported: S16.
     * @param[out] output   Output tensor. Data types supported: S16.
     * @param[in]  mag_type (Optional) Magnitude calculation type. Default: L2NORM.
     * @param[in]  use_fp16 (Optional) If true the FP16 kernels will be used. If false F32 kernels are used.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, MagnitudeType mag_type = MagnitudeType::L2NORM, bool use_fp16 = false);
};
}
#endif /*__ARM_COMPUTE_CLMAGNITUDE_H__ */
