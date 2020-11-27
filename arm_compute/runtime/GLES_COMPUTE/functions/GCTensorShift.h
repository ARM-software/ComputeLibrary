/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GCTENSORSHIFT_H
#define ARM_COMPUTE_GCTENSORSHIFT_H

#include "arm_compute/core/GLES_COMPUTE/kernels/GCTensorShiftKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

#include <memory>

namespace arm_compute
{
class IGCTensor;

/** Basic function to execute shift function for tensor. This function applies to fix alignment issue on OpenGL ES:
 *
 * @note This alignment issue is introduced by limits of compute shader which requires 32/64/128bit alignment for data access on OpenGL ES
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCTensorShift : public IGCSimpleFunction
{
public:
    /** Initialise the kernel's input, output.
     *
     * @param[in,out] input Source tensor. Data types supported: F16/F32.
     */
    void configure(IGCTensor *input);
};
}
#endif /* ARM_COMPUTE_GCTENSORSHIFT_H */
