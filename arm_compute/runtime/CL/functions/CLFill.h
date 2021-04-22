/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLFILL_H
#define ARM_COMPUTE_CLFILL_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;

/** Basic function to run @ref opencl::kernels::ClFillKernel */
class CLFill : public IFunction
{
public:
    /** Constructor */
    CLFill();
    /** Destructor */
    ~CLFill();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFill(const CLFill &) = delete;
    /** Default move constructor */
    CLFill(CLFill &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFill &operator=(const CLFill &) = delete;
    /** Default move assignment operator */
    CLFill &operator=(CLFill &&);
    /** Initialize the kernel's tensor and filling value
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in,out] tensor         Input tensor to fill. Supported data types: All.
     * @param[in]     constant_value The value used to fill the planes of the tensor
     * @param[in]     window         Window to be used in case setting only part of a tensor. Default is nullptr.
     */
    void configure(ICLTensor *tensor, const PixelValue &constant_value, Window *window = nullptr);
    /** Initialise the kernel's tensor and filling value
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] tensor          Input tensor to fill. Supported data types: All.
     * @param[in]     constant_value  The value used to fill the planes of the tensor
     * @param[in]     window          Window to be used in case setting only part of a tensor. Default is nullptr.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *tensor, const PixelValue &constant_value, Window *window = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFill
     *
     * @param[in] tensor         Source tensor info. Data types supported: All.
     * @param[in] constant_value The value used to fill the planes of the tensor
     * @param[in] window         Window to be used in case setting only part of a tensor. Default is nullptr.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *tensor, const PixelValue &constant_value, Window *window = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLFILL_H */
