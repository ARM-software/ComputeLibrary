/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCOPY_H
#define ARM_COMPUTE_CLCOPY_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref opencl::kernels::ClCopyKernel */
class CLCopy : public IFunction
{
public:
    /** Constructor */
    CLCopy();
    /** Destructor */
    ~CLCopy();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCopy(const CLCopy &) = delete;
    /** Default move constructor */
    CLCopy(CLCopy &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCopy &operator=(const CLCopy &) = delete;
    /** Default move assignment operator */
    CLCopy &operator=(CLCopy &&);
    /** Initialise the function's source and destination.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input      Source tensor. Data types supported: All.
     * @param[out] output     Output tensor. Data types supported: Same as @p input.
     * @param[in]  dst_window (Optional) Window to be used in case only copying into part of a tensor. Default is nullptr.
     */
    void configure(ICLTensor *input, ICLTensor *output, Window *dst_window = nullptr);
    /** Initialise the function's source and destination.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: All.
     * @param[out] output          Output tensor. Data types supported: Same as @p input.
     * @param[in]  dst_window      (Optional) Window to be used in case only copying into part of a tensor. Default is nullptr.
     *
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, Window *dst_window = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLCopy
     *
     * @param[in] input      Source tensor. Data types supported: All.
     * @param[in] output     Output tensor. Data types supported: Same as @p input.
     * @param[in] dst_window (Optional) Window to be used in case only copying into part of a tensor. Default is nullptr.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, Window *dst_window = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLCOPY_H */
