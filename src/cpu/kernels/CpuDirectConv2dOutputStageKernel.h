/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_DIRECT_CONV2D_OUTPUT_STAGE_KERNEL_H
#define ARM_COMPUTE_CPU_DIRECT_CONV2D_OUTPUT_STAGE_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel to accumulate the biases, if provided, or downscale in case of quantized input.
 *
 * @note We assume bias to be shared
 * @note For quantized computations (i.e. @p src of S32 type) the output data type for auto-initialization must be passed as part
 *       of the @ref DirectConvolutionLayerOutputStageKernelInfo.
 */
class CpuDirectConv2dOutputStageKernel : public ICpuKernel<CpuDirectConv2dOutputStageKernel>
{
public:
    CpuDirectConv2dOutputStageKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDirectConv2dOutputStageKernel);
    /** Set the accumulate buffer and the biases of the kernel.
     *
     * @param[in, out] src  Input to add the bias to. If @p dst is not specified then accumulation is done in-place.
     *                      Data type supported: F16/F32/S32
     * @param[in]      bias (Optional) The shared bias tensor to add. It must be 1D Tensor. Data type supported: Same as @p src
     * @param[out]     dst  (Optional) If the dst tensor is specified the accumulation is done out-of-place. (Defaults to nullptr)
     *                      Note that in-place computation is only supported for F16/F32. For S32 this must not be nullptr.
     *                      Data type supported: F16/F32 or QASYMM8/QASYMM8_SIGNED if @p src is S32
     * @param[in]      info (Optional) DirectConvolutionLayerOutputStageKernel descriptor metadata
     */
    void configure(ITensorInfo *src, const ITensorInfo *bias = nullptr, ITensorInfo *dst = nullptr,
                   const DirectConvolutionLayerOutputStageKernelInfo &info = DirectConvolutionLayerOutputStageKernelInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDirectConv2dOutputStageKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *bias = nullptr, const ITensorInfo *dst = nullptr,
                           const DirectConvolutionLayerOutputStageKernelInfo &info = DirectConvolutionLayerOutputStageKernelInfo());

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    using OutputStageKernel = void(ITensor *src, const ITensor *bias, const Window &window, ITensor *dst,
                                   int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift);

    OutputStageKernel *_func{ nullptr };
    int                _result_fixedpoint_multiplier{ 0 };
    int                _result_shift{ 0 };
    int                _result_offset_after_shift{ 0 };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DIRECT_CONV2D_OUTPUT_STAGE_KERNEL_H */
