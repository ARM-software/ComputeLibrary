/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLMINMAXLAYERKERNEL_H
#define ARM_COMPUTE_CLMINMAXLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to perform min max search on a 3D tensor.
 */
class CLMinMaxLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLMinMaxLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxLayerKernel(const CLMinMaxLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxLayerKernel &operator=(const CLMinMaxLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLMinMaxLayerKernel(CLMinMaxLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLMinMaxLayerKernel &operator=(CLMinMaxLayerKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor with at least 3 dimensions. The dimensions over the third will be interpreted as batches.Data types supported: F32.
     * @param[out] output Output tensor with shape [2, batches, ...] which stores the minimum and maximum values for each 3D input tensor.
     *                    The dimensions over the second must match the batched dimensions of the input tensor. Data types supported: F32.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor with at least 3 dimensions. The dimensions over the third will be interpreted as batches.Data types supported: F32.
     * @param[out] output          Output tensor with shape [2, batches, ...] which stores the minimum and maximum values for each 3D input tensor.
     *                    The dimensions over the second must match the batched dimensions of the input tensor. Data types supported: F32.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLMinMaxLayerKernel
     *
     * @param[in] input  Input tensor info.  Data types supported: F32.
     * @param[in] output Output tensor info with shape [2, batches, ...] which stores the minimum and maximum values for each 3D input tensor.
     *                   The dimensions over the second must match the batched dimensions of the input tensor. Data types supported: F32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    /** Resets global minimum and maximum
     *
     * @param[in,out] queue Command queue on which to map and unmap the min_max tensor
     */
    void reset(cl::CommandQueue &queue);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLMINMAXLAYERKERNEL_H */
