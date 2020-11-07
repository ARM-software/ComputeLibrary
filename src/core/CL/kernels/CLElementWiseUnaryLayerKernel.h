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
#ifndef ARM_COMPUTE_CLELEMENTWISEUNARYLAYERKERNEL_H
#define ARM_COMPUTE_CLELEMENTWISEUNARYLAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/ICLSimpleKernel.h"

namespace arm_compute
{
/** Interface for the elementwise unary operator */
class CLElementWiseUnaryLayerKernel : public ICLKernel
{
public:
    /** Initialise the kernel's inputs, output.
     *
     * @param[in]  input  First tensor input info. Data types supported: F16/F32.
     * @param[out] output Output tensor info. Data types supported: Same as @p input.
     * @param[in]  op     Element wise unary operation to perform.
     */
    void configure(const ITensorInfo *input, ITensorInfo *output, const ElementWiseUnary &op);
    /** Initialise the kernel's inputs, output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           First tensor input info. Data types supported: F16/F32.
     * @param[out] output          Output tensor info. Data types supported: Same as @p input.
     * @param[in]  op              Element wise unary operation to perform.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output, const ElementWiseUnary &op);
    /** Static function to check if given info will lead to a valid configuration of @ref CLElementWiseUnaryLayerKernel
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     * @param[in] op     Element wise unary operation to perform.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ElementWiseUnary &op);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLELEMENTWISEUNARYLAYERKERNEL_H */
