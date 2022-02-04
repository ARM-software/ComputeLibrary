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
#ifndef ARM_COMPUTE_CL_WINOGRAD_INPUT_TRANSFORM_KERNEL_H
#define ARM_COMPUTE_CL_WINOGRAD_INPUT_TRANSFORM_KERNEL_H

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
/** OpenCL kernel to perform Winograd input transform.*/
class ClWinogradInputTransformKernel : public IClKernel
{
public:
    ClWinogradInputTransformKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClWinogradInputTransformKernel);
    /** Set the input and output of the kernel.
     *
     * @note Winograd input transform supports the following configurations for NCWH data layout
     *       F(output tile, kernel size):F(2x2, 3x3), F(2x1, 3x1), F(1x2, 1x3),
     *                                   F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     * @note Winograd input transform supports the following configurations for NHWC data layout
     *       F(output tile, kernel size):F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     *       Strides: only unit strides
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] src             The input tensor info to transform. Data types supported: F16/F32
     * @param[in] dst             The output tensor info. The shape for this tensor can be calculated using the utility function @p compute_winograd_input_transform_shape. Data types supported: Same as @p input
     * @param[in] winograd_info   Contains Winograd's information described in @ref WinogradInfo.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const WinogradInfo &winograd_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClWinogradInputTransformKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const WinogradInfo &winograd_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    using WinogradKey = std::pair<std::pair<int, int>, std::pair<int, int>>;

    BorderSize   _border_size{ 0 };
    DataLayout   _data_layout{ DataLayout::UNKNOWN };
    int          _num_tiles_x{ 0 };
    int          _num_tiles_y{ 0 };
    unsigned int _step_z{ 1 };
    int32_t      _src_width{ 0 };
    int32_t      _src_height{ 0 };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_WINOGRAD_INPUT_TRANSFORM_KERNEL_H */
