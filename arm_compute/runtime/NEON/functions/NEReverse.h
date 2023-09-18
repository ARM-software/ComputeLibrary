/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEREVERSE_H
#define ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEREVERSE_H

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref NEReverseKernel */
class NEReverse : public INESimpleFunctionNoBorder
{
public:
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |All            |U32, S32       |All            |
     *
     * @param[in]  input             Input tensor. Data types supported: All
     * @param[out] output            Output tensor. Data type supported: Same as @p input
     * @param[in]  axis              Axis tensor. Contains the indices of the dimensions to reverse. Data type supported: U32/S32
     * @param[in]  use_inverted_axis Reverse ACL axis indices convention, if true, (inverted)axis = (tensor_rank - 1) - axis
     *
     * @note The value of each axis should be between [-rank, rank)
     * @note If there are duplicate values in the tensor, the subsequent axis values are ignored. e.g. an array of [2, 2] has the same effects as [2].
     *
     * @deprecated Support for U32 in axis tensor will be removed in 24.02 release
     *
     */
    void configure(const ITensor *input, ITensor *output, const ITensor *axis, const bool use_inverted_axis = false);
    /** Static function to check if given info will lead to a valid configuration of @ref NEReverseKernel
     *
     * @param[in] input             Input tensor info. Data types supported: All
     * @param[in] output            Output tensor info. Data type supported: Same as @p input
     * @param[in] axis              Axis tensor info. Contains the indices of the dimensions to reverse. Data type supported: U32/S32
     * @param[in] use_inverted_axis Reverse ACL axis indices convention, if true, (inverted)axis = (tensor_rank - 1) - axis
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis, const bool use_inverted_axis = false);
};
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEREVERSE_H
