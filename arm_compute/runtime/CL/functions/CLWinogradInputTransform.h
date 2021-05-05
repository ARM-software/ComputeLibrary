/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLWINOGRADINPUTTRANSFORM_H
#define ARM_COMPUTE_CLWINOGRADINPUTTRANSFORM_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to execute a @ref CLWinogradInputTransformKernel. */
class CLWinogradInputTransform : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
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
     * @param[in] input         The input tensor to transform. Data types supported: F16,F32
     * @param[in] output        The output tensor. The shape for this tensor can be calculated using the utility function @p compute_winograd_input_transform_shape. Data types supported: Same as @p input
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo.
     */
    void configure(ICLTensor *input, ICLTensor *output, const WinogradInfo &winograd_info);
    /** Set the input and output tensors.
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
     * @param[in] input           The input tensor to transform. Data types supported: F16,F32
     * @param[in] output          The output tensor. The shape for this tensor can be calculated using the utility function @p compute_winograd_input_transform_shape. Data types supported: Same as @p input
     * @param[in] winograd_info   Contains Winograd's information described in @ref WinogradInfo.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const WinogradInfo &winograd_info);
    /**  Static function to check if given info will lead to a valid configuration of @ref CLWinogradInputTransform.
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
     * @param[in] input         The input tensor to transform. Data types supported: F16,F32
     * @param[in] output        The output tensor. The shape for this tensor can be calculated using the utility function @p compute_winograd_input_transform_shape. Data types supported: Same as @p input
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info);
};
}
#endif /*ARM_COMPUTE_CLWINOGRADINPUTTRANSFORM_H */
