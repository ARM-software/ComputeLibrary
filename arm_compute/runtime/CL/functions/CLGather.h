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

#ifndef ARM_COMPUTE_CLGATHER_H
#define ARM_COMPUTE_CLGATHER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLGatherKernel */
class CLGather : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs and outputs
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input   Source tensor. Supported tensor rank: up to 4. Data type supported: All.
     * @param[in]  indices Indices tensor. Supported tensor rank: up to 1. Must be one of the following types: U32/S32. Each value must be in range [0, input.shape[@p axis]), otherwise the result will become unpredictable.
     * @param[out] output  Destination tensor. Data type supported: Same as @p input
     * @param[in]  axis    (Optional) The axis in @p input to gather @p indices from. Defaults to 0
     */
    void configure(const ICLTensor *input, const ICLTensor *indices, ICLTensor *output, int axis = 0);
    /** Initialise the kernel's inputs and outputs
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Supported tensor rank: up to 4. Data type supported: All.
     * @param[in]  indices         Indices tensor. Supported tensor rank: up to 1. Must be one of the following types: U32/S32. Each value must be in range [0, input.shape[@p axis]), otherwise the result will become unpredictable.
     * @param[out] output          Destination tensor. Data type supported: Same as @p input
     * @param[in]  axis            (Optional) The axis in @p input to gather @p indices from. Defaults to 0
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *indices, ICLTensor *output, int axis = 0);

    /** Static function to check if given info will lead to a valid configuration of @ref CLGatherKernel
     *
     * @param[in] input   Source tensor info. Supported tensor rank: up to 4. Data type supported: All.
     * @param[in] indices Indices tensor info. Supported tensor rank: up to 4. Must be one of the following types: U32/S32. Each value must be in range [0, input.shape[@p axis]), otherwise the result will become unpredictable.
     * @param[in] output  Destination tensor info. Data type supported: Same as @p input
     * @param[in] axis    (Optional) The axis in @p input to gather @p indices from. Defaults to 0
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis = 0);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLGATHER_H */
