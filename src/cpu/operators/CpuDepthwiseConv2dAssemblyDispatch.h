/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_ASSEMBLY_DISPATCH_H
#define ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_ASSEMBLY_DISPATCH_H

#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
struct ConvolutionInfo;

namespace cpu
{
/** Depthwise convolution assembly kernel glue */
class CpuDepthwiseConv2dAssemblyDispatch : public ICpuOperator
{
public:
    CpuDepthwiseConv2dAssemblyDispatch();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDepthwiseConv2dAssemblyDispatch);
    ~CpuDepthwiseConv2dAssemblyDispatch();
    /** Initialize the function's source, destination, kernels and border_size.
     *
     * @note Supports only NHWC format
     *
     * @param[in]  src     Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights Weights tensor info. These are 3D tensors with shape [W, H, IFM].
     *                     Data type supported: same as @p src or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p src is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  bias    (Optional) Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                     Data type supported: same as @p src or S32 if @p src is quantized.
     * @param[out] dst     Destination tensor info. Data type supported: same as @p src.
     * @param[in]  info    Depthwise convolution meta-data.
     */
    void configure(const ITensorInfo     *src,
                   const ITensorInfo     *weights,
                   const ITensorInfo     *bias,
                   ITensorInfo           *dst,
                   const ConvolutionInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDepthwiseConv2dAssemblyDispatch::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo     *src,
                           const ITensorInfo     *weights,
                           const ITensorInfo     *bias,
                           const ITensorInfo     *dst,
                           const ConvolutionInfo &info);
    /** Checks if activation is supported by the assembly kernels
     *
     * @param[in] activation Activation to check
     *
     * @return True if activation is supported else false
     */
    static bool is_activation_supported(const ActivationLayerInfo &activation);

    // Inherited methods overridden:
    void                             run(ITensorPack &tensors) override;
    void                             prepare(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    struct LocalImpl;
    std::unique_ptr<LocalImpl> _pImpl;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_ASSEMBLY_DISPATCH_H */
