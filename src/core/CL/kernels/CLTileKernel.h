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
#ifndef ARM_COMPUTE_CLTILEKERNEL_H
#define ARM_COMPUTE_CLTILEKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a Tile operation */
class CLTileKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLTileKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLTileKernel(const CLTileKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLTileKernel &operator=(const CLTileKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLTileKernel(CLTileKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLTileKernel &operator=(CLTileKernel &&) = default;
    /** Default destructor */
    ~CLTileKernel() = default;
    /** Set the source, destination of the kernel
     *
     * @param[in]  input     Source tensor. Data type supported: All.
     * @param[in]  multiples Contains the number of times the input tensor should be replicated on the given dimension.
     *                       Cannot have more than 4 elements (tiling in dimensions greater than 4 is not supported).
     * @param[out] output    Destination tensor. Same as @p input
     *
     */
    void configure(const ICLTensor *input, ICLTensor *output, const Multiples &multiples);
    /** Set the source, destination of the kernel
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data type supported: All.
     * @param[in]  multiples       Contains the number of times the input tensor should be replicated on the given dimension.
     *                             Cannot have more than 4 elements (tiling in dimensions greater than 4 is not supported).
     * @param[out] output          Destination tensor. Same as @p input
     *
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const Multiples &multiples);
    /** Static function to check if given info will lead to a valid configuration of @ref CLTileKernel
     *
     * @param[in] input     Source tensor info. Data type supported: All.
     * @param[in] multiples Contains the number of times the input tensor should be replicated on the given dimension.
     *                      Cannot have more than 4 elements (tiling in dimensions greater than 4 is not supported).
     * @param[in] output    Destination tensor info. Same as @p input
     *
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Multiples &multiples);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLTILEKERNEL_H */
