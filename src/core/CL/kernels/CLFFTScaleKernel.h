/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLFFTSCALEKERNEL_H
#define ARM_COMPUTE_CLFFTSCALEKERNEL_H

#include "src/core/CL/ICLKernel.h"

#include "arm_compute/core/KernelDescriptors.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Interface for the inverse fft scale kernel. */
class CLFFTScaleKernel : public ICLKernel
{
public:
    /** Constructor */
    CLFFTScaleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFFTScaleKernel(const CLFFTScaleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFFTScaleKernel &operator=(const CLFFTScaleKernel &) = delete;
    /** Default Move Constructor. */
    CLFFTScaleKernel(CLFFTScaleKernel &&) = default;
    /** Default move assignment operator */
    CLFFTScaleKernel &operator=(CLFFTScaleKernel &&) = default;
    /** Default destructor */
    ~CLFFTScaleKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in,out] input  Source tensor. Data types supported: F16/F32.
     * @param[out]    output Destination tensor. Data type supported: same as @p input
     * @param[in]     config Kernel configuration
     */
    void configure(ICLTensor *input, ICLTensor *output, const FFTScaleKernelInfo &config);
    /** Set the input and output tensors.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] input           Source tensor. Data types supported: F16/F32.
     * @param[out]    output          Destination tensor. Data type supported: same as @p input
     * @param[in]     config          Kernel configuration
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const FFTScaleKernelInfo &config);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFFTScaleKernel
     *
     * @param[in] input  Source tensor info. Data types supported: F16/F32.
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     * @param[in] config Kernel configuration
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const FFTScaleKernelInfo &config);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    ICLTensor *_input;
    ICLTensor *_output;
    bool       _run_in_place;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLFFTSCALEKERNEL_H */
