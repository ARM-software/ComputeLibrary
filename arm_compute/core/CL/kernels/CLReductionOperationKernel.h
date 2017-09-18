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
#ifndef __ARM_COMPUTE_CLREDUCTIONOPERATIONKERNEL_H__
#define __ARM_COMPUTE_CLREDUCTIONOPERATIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the reduction operation kernel */
class CLReductionOperationKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLReductionOperationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLReductionOperationKernel(const CLReductionOperationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLReductionOperationKernel &operator=(const CLReductionOperationKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLReductionOperationKernel(CLReductionOperationKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLReductionOperationKernel &operator=(CLReductionOperationKernel &&) = default;
    /** Default destructor */
    ~CLReductionOperationKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: F32.
     * @param[out] output Destination tensor. Data types supported: Same as @p input.
     * @param[in]  axis   Axis along which to reduce. Supported reduction axis : 0
     * @param[in]  op     Reduction operation to perform.
     */
    void configure(const ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor   *_input;
    ICLTensor         *_output;
    unsigned int       _reduction_axis;
    ReductionOperation _op;
    BorderSize         _border_size;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLREDUCTIONOPERATIONKERNEL_H__ */
