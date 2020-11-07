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
#ifndef ARM_COMPUTE_CLFLATTENLAYERKERNEL_H
#define ARM_COMPUTE_CLFLATTENLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL interface for the flatten kernel.*/
class CLFlattenLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLFlattenLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFlattenLayerKernel(const CLFlattenLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFlattenLayerKernel &operator=(const CLFlattenLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLFlattenLayerKernel(CLFlattenLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLFlattenLayerKernel &operator=(CLFlattenLayerKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input  First input tensor to flatten with at least 3 dimensions.
     *                    The dimensions above the third will be interpreted as batches. Data types supported: All.
     * @param[out] output Output tensor with shape [w*h*d, input_batches] where:
     *                    w = width input tensor, h = height input tensor and d = depth input tensor. Data type supported: same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Set the input and output of the kernel.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           First input tensor to flatten with at least 3 dimensions.
     *                             The dimensions above the third will be interpreted as batches. Data types supported: All.
     * @param[out] output          Output tensor with shape [w*h*d, input_batches] where:
     *                    w = width input tensor, h = height input tensor and d = depth input tensor. Data type supported: same as @p input
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFlattenLayerKernel
     *
     * @param[in]  input  First input tensor to flatten with at least 3 dimensions.
     *                    The dimensions above the third will be interpreted as batches. Data types supported: All.
     * @param[out] output Output tensor with shape [w*h*d, input_batches] where:
     *                    w = width input tensor, h = height input tensor and d = depth input tensor. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

public:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLFLATTENLAYERKERNEL_H */
