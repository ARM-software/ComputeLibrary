/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLACTIVATIONLAYERKERNEL_H
#define ARM_COMPUTE_CLACTIVATIONLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;
/** Interface for the activation layer kernel. */
class CLActivationLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLActivationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLActivationLayerKernel(const CLActivationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLActivationLayerKernel &operator=(const CLActivationLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLActivationLayerKernel(CLActivationLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLActivationLayerKernel &operator=(CLActivationLayerKernel &&) = default;
    /** Default destructor */
    ~CLActivationLayerKernel() = default;
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr, the activation function will be performed in-place
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input           Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out]     output          Destination tensor. Data type supported: same as @p input
     * @param[in]      act_info        Activation layer information.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *input, ITensorInfo *output, ActivationLayerInfo act_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLActivationLayerKernel
     *
     * @param[in] input    Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
     *                     of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[in] output   Destination tensor info. Data type supported: same as @p input
     * @param[in] act_info Activation layer information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    bool _run_in_place;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLACTIVATIONLAYERKERNEL_H */
