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
#ifndef __ARM_COMPUTE_CLWINOGRADFILTERTRANSFORMKERNEL_H__
#define __ARM_COMPUTE_CLWINOGRADFILTERTRANSFORMKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the Winograd filter transform kernel. */
class CLWinogradFilterTransformKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLWinogradFilterTransformKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradFilterTransformKernel(const CLWinogradFilterTransformKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradFilterTransformKernel &operator=(const CLWinogradFilterTransformKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLWinogradFilterTransformKernel(CLWinogradFilterTransformKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLWinogradFilterTransformKernel &operator=(CLWinogradFilterTransformKernel &&) = default;
    /** Default destructor */
    ~CLWinogradFilterTransformKernel() = default;
    /** Set the input and output tensor.
     *
     * @param[in]  input  Source tensor. The input is a 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM] (NCHW data layout).
     *                    kernel_x must be 3 and equal to kernel_y. Data types supported: F32.
     * @param[out] output Destination tensor. The output is a 3D tensor with dimensions [OFM, IFM, 16]. Data type supported: same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLWinogradFilterTransformKernel
     *
     * @param[in] input  Source tensor info. The input is a 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM] (NCHW data layout).
     *                   kernel_x must be 3 and equal to kernel_y. Data types supported: F32.
     * @param[in] output Destination tensor info. The output is a 3D tensor with dimensions [OFM, IFM, 16]. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLWINOGRADFILTERTRANSFORMKERNEL_H__ */
