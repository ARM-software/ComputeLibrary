/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLWINOGRADINPUTTRANSFORMKERNEL_H__
#define __ARM_COMPUTE_CLWINOGRADINPUTTRANSFORMKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform Winograd input transform.*/
class CLWinogradInputTransformKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLWinogradInputTransformKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradInputTransformKernel(const CLWinogradInputTransformKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradInputTransformKernel &operator=(const CLWinogradInputTransformKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLWinogradInputTransformKernel(CLWinogradInputTransformKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLWinogradInputTransformKernel &operator=(CLWinogradInputTransformKernel &&) = default;
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
     * @param[in] input         The input tensor to transform. Data types supported: F16/F32
     * @param[in] output        The output tensor. The shape for this tensor can be calculated using the utility function @p compute_winograd_input_transform_shape. Data types supported: Same as @p input
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const WinogradInfo &winograd_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLWinogradInputTransformKernel
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
     * @param[in] input         The input tensor to transform. Data types supported: F16/F32
     * @param[in] output        The output tensor. The shape for this tensor can be calculated using the utility function @p compute_winograd_input_transform_shape. Data types supported: Same as @p input
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    using WinogradKey = std::pair<std::pair<int, int>, std::pair<int, int>>;

    BorderSize       _border_size;
    const ICLTensor *_input;
    ICLTensor       *_output;
    DataLayout       _data_layout;
    int              _num_tiles_x;
    int              _num_tiles_y;
    unsigned int     _step_z;
};
} // arm_compute
#endif /*__ARM_COMPUTE_CLWINOGRADINPUTTRANSFORMKERNEL_H__ */
