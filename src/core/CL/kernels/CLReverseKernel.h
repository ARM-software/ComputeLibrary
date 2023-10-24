/*
 * Copyright (c) 2018-2020, 2023 Arm Limited.
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
#ifndef ACL_SRC_CORE_CL_KERNELS_CLREVERSEKERNEL_H
#define ACL_SRC_CORE_CL_KERNELS_CLREVERSEKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the reverse kernel */
class CLReverseKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLReverseKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLReverseKernel(const CLReverseKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLReverseKernel &operator=(const CLReverseKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLReverseKernel(CLReverseKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLReverseKernel &operator=(CLReverseKernel &&) = default;
    /** Default destructor */
    ~CLReverseKernel() = default;
    /** Initialise the kernel's inputis and output
     *
     * @param[in]  input             Input tensor. Data types supported: All.
     * @param[out] output            Output tensor. Data type supported: Same as @p input
     * @param[in]  axis              Axis tensor. Contains the indices of the dimensions to reverse. Data type supported: U32/S32
     * @param[in]  use_inverted_axis Reverse ACL axis indices convention i.e. acl.dim(0) = tensor_rank -1
     *
     * @note The value of each axis should be between [-rank, rank)
     * @note If there are duplicate values in the tensor, the subsequent axis values are ignored. e.g. an array of [2, 2] has the same effects as [2].
     *
     * @deprecated Support for U32 in axis tensor will be removed in 24.02 release
     *
     */
    void configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *axis, bool use_inverted_axis);
    /** Initialise the kernel's inputis and output
     *
     * @param[in]  compile_context   The compile context to be used.
     * @param[in]  input             Input tensor. Data types supported: All.
     * @param[out] output            Output tensor. Data type supported: Same as @p input
     * @param[in]  axis              Axis tensor. Contains the indices of the dimensions to reverse. Data type supported: U32/S32
     * @param[in]  use_inverted_axis Reverse ACL axis indices convention i.e. acl.dim(0) = tensor_rank -1
     */
    void configure(const CLCompileContext &compile_context,
                   const ICLTensor        *input,
                   ICLTensor              *output,
                   const ICLTensor        *axis,
                   bool                    use_inverted_axis);

    /** Static function to check if given info will lead to a valid configuration of @ref CLReverseKernel
     *
     * @param[in] input             Input tensor info. Data types supported: All.
     * @param[in] output            Output tensor info. Data type supported: Same as @p input
     * @param[in] axis              Axis tensor info. Contains the indices of the dimensions to reverse. Data type supported: U32/S32
     * @param[in] use_inverted_axis Reverse ACL axis indices convention i.e. acl.dim(0) = tensor_rank -1
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis, bool use_inverted_axis);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

public:
    const ICLTensor *_input;
    ICLTensor       *_output;
    const ICLTensor *_axis;
};
} // namespace arm_compute
#endif // ACL_SRC_CORE_CL_KERNELS_CLREVERSEKERNEL_H
