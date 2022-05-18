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
#ifndef ARM_COMPUTE_CPU_DIRECT_CONV2D_KERNEL_H
#define ARM_COMPUTE_CPU_DIRECT_CONV2D_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to perform Direct Convolution Layer. */
class CpuDirectConv2dKernel : public ICpuKernel<CpuDirectConv2dKernel>
{
private:
    using DirectConv2dKernel_Ptr = std::add_pointer<void(const Window &, const ITensor *, const ITensor *, ITensor *, const PadStrideInfo &)>::type;

public:
    CpuDirectConv2dKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDirectConv2dKernel);
    /** Set the src, weights, and dst tensors.
     *
     * @note: DirectConvolution only works in the following configurations:
     *        1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3
     *        3x3 convolution with stride_x = 1/2/3, stride_y = 1/2/3
     *
     * @param[in]  src       The input tensor to convolve. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F16/F32.
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                       The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                       Data type supported:Same as @p input.
     * @param[out] dst       Output tensor.
     *                       The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: F16/F32
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(ITensorInfo *src, ITensorInfo *weights, ITensorInfo *dst, const PadStrideInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDirectConv2dKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct DirectConv2dKernel
    {
        const char                         *name;
        const DataTypeDataLayoutSelectorPtr is_selected;
        DirectConv2dKernel_Ptr              ukernel;
    };

    static const std::vector<DirectConv2dKernel> &get_available_kernels();

private:
    PadStrideInfo _conv_info{};
    unsigned int  _kernel_size{ 0 };
    DataLayout    _data_layout{ DataLayout::UNKNOWN };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_DIRECTCONV2D_KERNEL_H */
