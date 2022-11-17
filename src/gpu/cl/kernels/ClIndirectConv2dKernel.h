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
#ifndef ARM_COMPUTE_CL_DIRECT_CONV2D_KERNEL_H
#define ARM_COMPUTE_CL_DIRECT_CONV2D_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
// Forward declaration
struct DirectConvComputeKernelInfo;

namespace opencl
{
namespace kernels
{
/** Interface for the  indirect convolution kernel. */
class ClIndirectConv2dKernel : public IClKernel
{
public:
    ClIndirectConv2dKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClIndirectConv2dKernel);
    /** Set the src, offset, weights, biases and dst tensors info.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             The src tensor info to convolve. 3 lower dimensions represent a single src [IFM, width, height],
     *                             while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F16/F32.
     * @param[in]  off             The indirect buffer tensor info. Data types supported: S32.
     * @param[in]  weights         Weights tensor info. Weights are 4D tensor with dimensions [IFM, kernel_x, kernel_y, OFM].
     *                             Data type supported: Same as @p src.
     * @param[in]  biases          Biases tensor info. Biases are 1D tensor with dimension [OFM].
     *                             Data type supported: Same as @p src.
     * @param[out] dst             Output tensor info.
     *                             The 3rd dimension must be equal to the 4th dimension of the @p weights tensor. Data types supported: Same as @p src.
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info        Contains activaton information described in @ref ActivationLayerInfo.
     * @param[in]  desc            Direct convolution descriptor used to build the NHWC indirect convolution kernel.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *off, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                   const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClIndirectConv2dKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *off, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                           const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

public:
    bool _export_to_cl_image{ false };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_DIRECT_CONV2D_KERNEL_H */
