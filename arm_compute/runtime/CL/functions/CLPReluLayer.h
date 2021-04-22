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
#ifndef ARM_COMPUTE_CLPRELULAYER_H
#define ARM_COMPUTE_CLPRELULAYER_H

#include "arm_compute/runtime/CL/ICLOperator.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;
/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for PRELU
 *
 * @note The function implements an activation layer with the PRELU activation function.
 */
class CLPReluLayer : public IFunction
{
public:
    /** Default Constructor */
    CLPReluLayer();
    /** Default Destructor */
    ~CLPReluLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPReluLayer(const CLPReluLayer &) = delete;
    /** Default move constructor */
    CLPReluLayer(CLPReluLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPReluLayer &operator=(const CLPReluLayer &) = delete;
    /** Default move assignment operator */
    CLPReluLayer &operator=(CLPReluLayer &&);
    /** Set the input and output tensor.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  alpha  PRelu layer parameters. Data types supported: same of @p input.
     * @param[out] output Destination tensor. Data type supported: same as @p input
     */
    void configure(ICLTensor *input, ICLTensor *alpha, ICLTensor *output);
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  alpha           PRelu layer parameters. Data types supported: same of @p input.
     * @param[out] output          Destination tensor. Data type supported: same as @p input
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *alpha, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPReluLayer
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] alpha  PRelu layer parameters. Data types supported: same of @p input.
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *alpha, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLPRELULAYER_H */
