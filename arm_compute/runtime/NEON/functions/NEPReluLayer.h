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
#ifndef ARM_COMPUTE_NEPRELULAYER_H
#define ARM_COMPUTE_NEPRELULAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::kernels::CpuArithmeticKernel for PRELU
 *
 * @note The function implements an activation layer with the PRELU activation function.
 */
class NEPReluLayer : public IFunction
{
public:
    /** Default Constructor */
    NEPReluLayer();
    /** Default Destructor */
    ~NEPReluLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPReluLayer(const NEPReluLayer &) = delete;
    /** Default move constructor */
    NEPReluLayer(NEPReluLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPReluLayer &operator=(const NEPReluLayer &) = delete;
    /** Default move assignment operator */
    NEPReluLayer &operator=(NEPReluLayer &&);
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
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  alpha  Source alpha tensor. Data types supported: same of @p input.
     * @param[out] output Destination tensor. Data type supported: same as @p input
     */
    void configure(const ITensor *input, const ITensor *alpha, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPReluLayer
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] alpha  Source alpha tensor info. Data types supported: same of @p input.
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
#endif /* ARM_COMPUTE_NEPRELULAYER_H */
