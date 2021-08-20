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
#ifndef ARM_COMPUTE_CL_ACTIVATION_KERNEL_H
#define ARM_COMPUTE_CL_ACTIVATION_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the activation kernel. */
class ClActivationKernel : public IClKernel
{
public:
    ClActivationKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClActivationKernel);
    /** Configure kernel for a given list of arguments
     *
     * @note If the output tensor is a nullptr, the activation function will be performed in-place
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] src             Source tensor info. In case of @p dst tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out]     dst             Destination tensor info. Data type supported: same as @p src
     * @param[in]      act_info        Activation layer information.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, ActivationLayerInfo act_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClActivationKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &act_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue) override;

private:
    bool _run_in_place{ false };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_ACTIVATION_KERNEL_H */
