/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_WINOGRAD_OUTPUT_TRANSFORM_KERNEL_H
#define ARM_COMPUTE_CL_WINOGRAD_OUTPUT_TRANSFORM_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the Winograd output transform kernel. */
class ClWinogradOutputTransformKernel : public IClKernel
{
public:
    ClWinogradOutputTransformKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClWinogradOutputTransformKernel);
    /** Set the input and output tensor.
     *
     * @note Winograd output transform supports the following configurations for NCWH data layout
     *       F(output tile, kernel size):F(2x2, 3x3), F(2x1, 3x1), F(1x2, 1x3),
     *                                   F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     * @note Winograd output transform supports the following configurations for NHWC data layout
     *       F(output tile, kernel size):F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     *       Strides: only unit strides
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info with shape [C, N, K, batches]. Data types supported: F16/F32.
     * @param[in]  bias            Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. It can be a nullptr. Data type supported: as @p src
     * @param[out] dst             The output tensor info. The shape for this tensor can be calculated using the utility function @p compute_winograd_output_transform_shape. Data types supported: Same as @p src
     * @param[in]  winograd_info   Contains Winograd's information described in @ref WinogradInfo
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *bias, ITensorInfo *dst, const WinogradInfo &winograd_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClWinogradOutputTransformKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    using WinogradKey = std::pair<std::pair<int, int>, std::pair<int, int>>;

    bool    _is_nhwc{ false };
    int32_t _src_height{ 0 };
    int32_t _dst_width{ 0 };
    int32_t _dst_height{ 0 };
    int32_t _num_tiles_x{ 0 };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_WINOGRAD_OUTPUT_TRANSFORM_KERNEL_H */
