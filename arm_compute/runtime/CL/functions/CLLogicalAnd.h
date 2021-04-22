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
#ifndef ARM_COMPUTE_CLLOGICALAND_H
#define ARM_COMPUTE_CLLOGICALAND_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/ICLOperator.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref arm_compute::opencl::kernels::ClLogicalBinaryKernel.
 *
 * @note The tensor data type for the inputs must be U8.
 * @note The function performs a logical AND operation using the two input tensors.
 */
class CLLogicalAnd : public IFunction
{
public:
    /** Default Constructor */
    CLLogicalAnd();
    /** Default Destructor */
    ~CLLogicalAnd();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogicalAnd(const CLLogicalAnd &) = delete;
    /** Default move constructor */
    CLLogicalAnd(CLLogicalAnd &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogicalAnd &operator=(const CLLogicalAnd &) = delete;
    /** Default move assignment operator */
    CLLogicalAnd &operator=(CLLogicalAnd &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1          |dst          |
     * |:--------------|:-------------|:------------|
     * |U8             |U8            |U8           |
     *
     * @param[in]  input1 Input tensor. Data types supported: U8.
     * @param[in]  input2 Input tensor. Data types supported: same as @p input1.
     * @param[out] output Output tensor. Data types supported: same as @p input1.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          Input tensor. Data types supported: U8.
     * @param[in]  input2          Input tensor. Data types supported: same as @p input1.
     * @param[out] output          Output tensor. Data types supported: same as @p input1.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * @param[in] input1 First tensor input info. Data types supported: U8.
     * @param[in] input2 Second tensor input info. Data types supported: same as @p input1.
     * @param[in] output Output tensor info. Data types supported: same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

namespace experimental
{
class CLLogicalAnd : public ICLOperator
{
public:
    /** Default Constructor */
    CLLogicalAnd() = default;
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: U8.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Data types supported: same as @p input1.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: same as @p input1.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref arm_compute::opencl::kernels::ClLogicalBinaryKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: U8.
     * @param[in] input2 Second tensor input info. Data types supported: same as @p input1.
     * @param[in] output Output tensor info. Data types supported: same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLLOGICALAND_H */
