/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLCOPYKERNEL_H__
#define __ARM_COMPUTE_CLCOPYKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a copy between two tensors */
class CLCopyKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLCopyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLCopyKernel(const CLCopyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLCopyKernel &operator=(const CLCopyKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLCopyKernel(CLCopyKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLCopyKernel &operator=(CLCopyKernel &&) = default;
    /** Initialize the kernel's input, output.
     *
     * @param[in]  input         Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output        Destination tensor. Data types supported: same as @p input.
     * @param[in]  padding       (Optional) Padding to be applied to the input tensor
     * @param[in]  output_window (Optional) Window to be used in case only copying into part of a tensor. Default is nullptr.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const PaddingList &padding = PaddingList(), Window *output_window = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLCopyKernel
     *
     * @param[in] input         Source tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] output        Destination tensor info. Data types supported: same as @p input.
     * @param[in] padding       (Optional) Padding to be applied to the input tensor
     * @param[in] output_window (Optional) Window to be used in case only copying into part of a tensor. Default is nullptr.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding = PaddingList(), Window *output_window = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    Window           _output_window;
    bool             _has_output_window;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLCOPYKERNEL_H__ */
