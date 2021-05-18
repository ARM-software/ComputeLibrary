/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLLOGICALNOT_H
#define ARM_COMPUTE_CLLOGICALNOT_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to do logical NOT operation
 *
 * @note The tensor data type for the inputs must be U8.
 * @note The function performs a logical NOT operation on input tensor.
 */
class CLLogicalNot : public IFunction
{
public:
    /** Default Constructor */
    CLLogicalNot();
    /** Default Destructor */
    ~CLLogicalNot();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogicalNot(const CLLogicalNot &) = delete;
    /** Default move constructor */
    CLLogicalNot(CLLogicalNot &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogicalNot &operator=(const CLLogicalNot &) = delete;
    /** Default move assignment operator */
    CLLogicalNot &operator=(CLLogicalNot &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst          |
     * |:--------------|:------------|
     * |U8             |U8           |
     *
     * @param[in]  input  Input tensor. Data types supported: U8.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: U8.
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * @param[in] input  Tensor input info. Data types supported: U8.
     * @param[in] output Output tensor info. Data types supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLLOGICALNOT_H */
