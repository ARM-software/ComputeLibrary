/*
 * Copyright (c) 2017-2020, 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_ICLTUNER_H
#define ARM_COMPUTE_ICLTUNER_H

#include "arm_compute/core/experimental/Types.h"

namespace arm_compute
{
class ICLKernel;

/** Basic interface for tuning the OpenCL kernels */
class ICLTuner
{
public:
    /** Virtual destructor */
    virtual ~ICLTuner() = default;
    /** Tune OpenCL kernel statically
     *
     * @note Tuning is performed using only kernel and tensor metadata,
     *       thus can be performed when memory is not available
     *
     * @param[in] kernel Kernel to tune
     */
    virtual void tune_kernel_static(ICLKernel &kernel) = 0;
    /** Tune OpenCL kernel dynamically
     *
     * @note Tuning requires memory to be available on all kernel tensors and objects in order to be performed
     *
     * @param[in] kernel Kernel to tune
     */
    virtual void tune_kernel_dynamic(ICLKernel &kernel) = 0;
    /** Tune OpenCL kernel dynamically
     *
     * @param[in]      kernel  Kernel to tune
     * @param[in, out] tensors Tensors for the kernel to use
     */
    virtual void tune_kernel_dynamic(ICLKernel &kernel, ITensorPack &tensors) = 0;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_ICLTUNER_H */
