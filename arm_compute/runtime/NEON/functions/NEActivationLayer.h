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
#ifndef ARM_COMPUTE_NEACTIVATIONLAYER_H
#define ARM_COMPUTE_NEACTIVATIONLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IRuntimeContext.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::kernels::CpuActivationKernel
 *
 * @note The function simulates an activation layer with the specified activation function.
 */
class NEActivationLayer : public IFunction
{
public:
    /** Constructor
     *
     * @param[in] ctx Runtime context to be used by the function
     */
    NEActivationLayer(IRuntimeContext *ctx = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEActivationLayer(const NEActivationLayer &) = delete;
    /** Default move constructor */
    NEActivationLayer(NEActivationLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEActivationLayer &operator=(const NEActivationLayer &) = delete;
    /** Default move assignment operator */
    NEActivationLayer &operator=(NEActivationLayer &&);
    /** Destructor */
    ~NEActivationLayer();
    /** [NEActivationLayer snippet] **/
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
     * |QSYMM16        |QSYMM16        |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in, out] input           Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out]     output          Destination tensor. Data type supported: same as @p input
     * @param[in]      activation_info Activation layer parameters.
     */
    void configure(ITensor *input, ITensor *output, ActivationLayerInfo activation_info);
    /** [NEActivationLayer snippet] **/
    /** Static function to check if given info will lead to a valid configuration of @ref NEActivationLayer
     *
     * @param[in] input    Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
     *                     of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[in] output   Destination tensor info. Data type supported: same as @p input
     * @param[in] act_info Activation layer information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_computes
#endif /* ARM_COMPUTE_NEACTIVATIONLAYER_H */
