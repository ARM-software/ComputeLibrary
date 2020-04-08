/*
 * Copyright (c) 2018-2020 ARM Limited.
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

#ifndef ARM_COMPUTE_CLWIDTHCONCATENATE_2TENSORS_KERNEL_H
#define ARM_COMPUTE_CLWIDTHCONCATENATE_2TENSORS_KERNEL_H

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the width concatenate kernel of 2 tensors.
 *  The input1 and input2 tensors will be concatenated into the output tensor.
 */
class CLWidthConcatenate2TensorsKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLWidthConcatenate2TensorsKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWidthConcatenate2TensorsKernel(const CLWidthConcatenate2TensorsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWidthConcatenate2TensorsKernel &operator=(const CLWidthConcatenate2TensorsKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLWidthConcatenate2TensorsKernel(CLWidthConcatenate2TensorsKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLWidthConcatenate2TensorsKernel &operator=(CLWidthConcatenate2TensorsKernel &&) = default;
    /** Default destructor */
    ~CLWidthConcatenate2TensorsKernel() = default;
    /** Initialise the kernel's input1s and output
     *
     * @param[in]  input1 First input tensor. Data types supported: All.
     * @param[in]  input2 Second input tensor. Data types supported: same as @p input1
     * @param[out] output Output tensor. Data types supported: Same as @p input1.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output);
    /** Initialise the kernel's input1s and output
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          First input tensor. Data types supported: All.
     * @param[in]  input2          Second input tensor. Data types supported: same as @p input1
     * @param[out] output          Output tensor. Data types supported: Same as @p input1.
     */
    void configure(CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output);
    /**  Static function to check if given info will lead to a valid configuration of @ref CLWidthConcatenate2TensorsKernel
     *
     * @param[in] input1 First tensor info. Data types supported: All.
     * @param[in] input2 Second tensor info. Data types supported: same as @p input1
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input1;
    const ICLTensor *_input2;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLWIDTHCONCATENATE_2TENSORS_KERNEL_H */
