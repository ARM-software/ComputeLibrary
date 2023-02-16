/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_INDIRECT_CONV2D_ADDRESS_PRECALCULATION_KERNEL_H
#define ARM_COMPUTE_CL_INDIRECT_CONV2D_ADDRESS_PRECALCULATION_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
// Forward declarations
struct DirectConvComputeKernelInfo;

namespace opencl
{
namespace kernels
{
/** Interface for the  direct convolution kernel. */
class ClIndirectConv2dAddressPrecalculationKernel : public IClKernel
{
public:
    ClIndirectConv2dAddressPrecalculationKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClIndirectConv2dAddressPrecalculationKernel);
    /** Set the src, weights, biases and dst tensors info.
     *
     * @note: When M0 is 5,6,7, the kernel rounds up M0 to the nearest power of two. Therefore, eight. The reason behind
     *        this implementation detail is because we can exploit native opencl stores in the kernel.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             The src tensor info to convolve. 3 lower dimensions represent a single src [IFM, width, height],
     *                             while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F16/F32.
     * @param[in]  weights         Weights tensor info. Weights are 4D tensor with dimensions [IFM, kernel_x, kernel_y, OFM].
     *                             The 1st dimension must be the same as the src's volume 1st dimension.
     *                             Data type supported:Same as @p src.
     * @param[out] dst             Output tensor info where to store the precalculated offsets. Data types supported: S32.
     *                             The output is a 3D tensor with the following dimensions: [M0 x Kw x Kh, ceil(M/M0), batch-size], where:
     *                             Kw=Kernel width, Kh=Kernel height, M0=number of rows processed by each workitem, and M=dst_width x dst_height.
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  desc            Direct convolution descriptor used to build the NHWC direct/indirect convolution kernel.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *dst,
                   const PadStrideInfo &conv_info, const DirectConvComputeKernelInfo &desc);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClIndirectConv2dAddressPreCalculationKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst,
                           const PadStrideInfo &conv_info, const DirectConvComputeKernelInfo &desc);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_INDIRECT_CONV2D_ADDRESS_PRECALCULATION_KERNEL_H */
