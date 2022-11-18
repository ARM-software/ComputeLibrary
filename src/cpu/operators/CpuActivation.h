/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_ACTIVATION_H
#define ARM_COMPUTE_CPU_ACTIVATION_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuActivationKernel */
class CpuActivation : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  input           Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out] output          Destination tensor info. Data type supported: same as @p src
     * @param[in]  activation_info Activation layer parameters.
     */
    void configure(const ITensorInfo *input, ITensorInfo *output, const ActivationLayerInfo &activation_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuActivation::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ACTIVATION_H */
