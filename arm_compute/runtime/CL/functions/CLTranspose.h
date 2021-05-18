/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLTRANSPOSE_H
#define ARM_COMPUTE_CLTRANSPOSE_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to execute an @ref opencl::kernels::ClTransposeKernel. */
class CLTranspose : public IFunction
{
public:
    /** Constructor */
    CLTranspose();
    /** Destructor */
    ~CLTranspose();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLTranspose(const CLTranspose &) = delete;
    /** Default move constructor */
    CLTranspose(CLTranspose &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLTranspose &operator=(const CLTranspose &) = delete;
    /** Default move assignment operator */
    CLTranspose &operator=(CLTranspose &&) = default;
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input  Input tensor. Data types supported: All.
     * @param[out] output Output tensor. Data type supported: Same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: All.
     * @param[out] output          Output tensor. Data type supported: Same as @p input
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLTranspose
     *
     * @param[in] input  The input tensor. Data types supported: All.
     * @param[in] output The output tensor. Data types supported: Same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}

#endif /* ARM_COMPUTE_CLTRANSPOSE_H */
