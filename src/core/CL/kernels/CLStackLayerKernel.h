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

#ifndef ARM_COMPUTE_CLSTACKLAYERKERNEL_H
#define ARM_COMPUTE_CLSTACKLAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to stacks a rank-R tensor into one with rank-(R+1) along the axis dimension.*/
class CLStackLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLStackLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLStackLayerKernel(const CLStackLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLStackLayerKernel &operator=(const CLStackLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLStackLayerKernel(CLStackLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLStackLayerKernel &operator=(CLStackLayerKernel &&) = default;
    /** Default destructor */
    ~CLStackLayerKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in]  input       Input tensor. Data types supported: All.
     * @param[in]  axis        The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     * @param[in]  idx_input   Index of the input tensor in the list of tensors to stack.
     *                         All tensors in the list must have the same shape
     * @param[in]  num_tensors Number of tensors to stack
     * @param[out] output      Output tensor. Data types supported: Same as @p input.
     *
     */
    void configure(const ICLTensor *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, ICLTensor *output);
    /** Initialise the kernel's inputs and output
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: All.
     * @param[in]  axis            The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     * @param[in]  idx_input       Index of the input tensor in the list of tensors to stack.
     *                             All tensors in the list must have the same shape
     * @param[in]  num_tensors     Number of tensors to stack
     * @param[out] output          Output tensor. Data types supported: Same as @p input.
     *
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLStackLayerKernel
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in] input       Input tensor info. Data types supported: All.
     * @param[in] axis        The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     * @param[in] idx_input   Index of the input tensor in the list of tensors to stack
     *                        All tensors in the list must have the same shape
     * @param[in] num_tensors Number of tensors to stack
     * @param[in] output      Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLSTACKLAYERKERNEL_H */
