/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCONCATENATE_H
#define ARM_COMPUTE_CLCONCATENATE_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"

#include <vector>

namespace arm_compute
{
namespace opencl
{
/** Basic function to execute concatenate tensors along a given axis. This function calls the following kernels:
 *
 * -# @ref kernels::ClWidthConcatenateKernel (if underlying concatenation axis is 0).
 * -# @ref kernels::ClHeightConcatenateKernel (if underlying concatenation axis is 1).
 * -# @ref kernels::ClDepthConcatenateKernel (if underlying concatenation axis is 2).
 * -# @ref kernels::ClBatchConcatenateKernel (if underlying concatenation axis is 3).
 */
class ClConcatenate : public IClOperator
{
public:
    ClConcatenate() = default;
    /** Initialise the kernel's inputs vector and dst.
     *
     * @note Input and dst tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref kernels::ClWidthConcatenateKernel,
     *       @ref kernels::ClHeightConcatenateKernel and @ref kernels::ClDepthConcatenateKernel.
     *
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] src_vector      The vectors containing all the tensors info to concatenate. Data types supported: All
     * @param[out]    dst             Destination tensor info. Data types supported: same as @p src_vector.
     * @param[in]     axis            Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     */
    void configure(const ClCompileContext &compile_context, const std::vector<ITensorInfo *> &src_vector, ITensorInfo *dst, size_t axis);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClConcatenate::configure()
     *
     * @return a status
     */
    static Status validate(const std::vector<const ITensorInfo *> &src_vector, const ITensorInfo *dst, size_t axis);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::vector<std::unique_ptr<IClKernel>> _concat_kernels{};
    unsigned int                            _num_inputs{ 0 };
    unsigned int                            _axis{ 0 };
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_CONCATENATE_H */
