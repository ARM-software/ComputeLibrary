/*
 * Copyright (c) 2019-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_NATIVE_KERNEL_H
#define ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_NATIVE_KERNEL_H

#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"
#include "support/AclRequires.h"

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <arm_neon.h>
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to run a depthwise convolution native on a tensor. */
class CpuDepthwiseConv2dNativeKernel : public ICpuKernel<CpuDepthwiseConv2dNativeKernel>
{
private:
    using DepthwiseConv2dNativeKernelPtr =
        std::add_pointer<void(const ITensor *, const ITensor *, const ITensor *, ITensor *, const Window &, bool, const ConvolutionInfo &)>::type;

public:
    CpuDepthwiseConv2dNativeKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDepthwiseConv2dNativeKernel);

    /** Initialize the function's source, destination and parameters.
     *
     * @note Supported data layouts: NHWC
     *
     * @param[in]  src     Source tensor. DataType supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights Weights tensor. This is a 3D tensor with dimensions [IFM, W, H].
     *                     Data type supported: Same as @p src or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p src is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  biases  Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                     Data type supported: Same as @p src, S32 when src is QASYMM8/QASYMM8_SIGNED.
     * @param[out] dst     Destination tensor. Data type supported: Same as @p src.
     * @param[in]  info    Depthwise convolution meta-data.
     *
     */
    void configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ConvolutionInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDepthwiseConv2dNativeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;
    struct DepthwiseConv2dNativeKernel
    {
        const char                                       *name;
        const DepthwiseConv2dNativeDataTypeISASelectorPtr is_selected;
        DepthwiseConv2dNativeKernelPtr                    ukernel;
    };
    static const std::vector<DepthwiseConv2dNativeKernel> &get_available_kernels();

private:
    /** Common signature for all the specialised depthwise convolution native functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    DepthwiseConv2dNativeKernelPtr _func{ nullptr };
    ConvolutionInfo                _conv_info{};
    bool                           _has_biases{ false };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_NATIVE_KERNEL_H */
