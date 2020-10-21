/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCOMPARISONKERNEL_H
#define ARM_COMPUTE_CLCOMPARISONKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Interface for the comparison kernel. */
class CLComparisonKernel : public ICLKernel
{
public:
    /** Default constructor. */
    CLComparisonKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComparisonKernel(const CLComparisonKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComparisonKernel &operator=(const CLComparisonKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLComparisonKernel(CLComparisonKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLComparisonKernel &operator=(CLComparisonKernel &&) = default;
    /** Default destructor */
    ~CLComparisonKernel() = default;
    /** Set the inputs and output tensors
     *
     * @param[in]  input1    Source tensor. Data types supported: All.
     * @param[in]  input2    Source tensor. Data types supported: Same as @p input1.
     * @param[out] output    Destination tensor. Data types supported: U8.
     * @param[in]  operation Comparison operation to use.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ComparisonOperation operation);
    /** Set the inputs and output tensors
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          Source tensor. Data types supported: All.
     * @param[in]  input2          Source tensor. Data types supported: Same as @p input1.
     * @param[out] output          Destination tensor. Data types supported: U8.
     * @param[in]  operation       Comparison operation to use.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ComparisonOperation operation);
    /** Static function to check if given info will lead to a valid configuration of @ref CLComparisonKernel
     *
     * @param[in] input1    Source tensor. Data types supported: All.
     * @param[in] input2    Source tensor. Data types supported: Same as @p input1.
     * @param[in] output    Destination tensor. Data types supported: U8.
     * @param[in] operation Comparison operation to use.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation operation);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input1; /**< Source tensor 1 */
    const ICLTensor *_input2; /**< Source tensor 2 */
    ICLTensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLCOMPARISONKERNEL_H */
