/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLFLOORKERNEL_H__
#define __ARM_COMPUTE_CLFLOORKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a floor operation */
class CLFloorKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLFloorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFloorKernel(const CLFloorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFloorKernel &operator=(const CLFloorKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLFloorKernel(CLFloorKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLFloorKernel &operator=(CLFloorKernel &&) = default;
    /** Default destructor */
    ~CLFloorKernel() = default;
    /** Set the source, destination of the kernel
     *
     * @param[in]  input  Source tensor. Data type supported: F32.
     * @param[out] output Destination tensor. Data type supported: F32.
     */
    void configure(const ICLTensor *input, ICLTensor *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLFLOORKERNEL_H__ */
