/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEACTIVATIONLAYERKERNEL_H__
#define __ARM_COMPUTE_NEACTIVATIONLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/utils/misc/Traits.h"

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <arm_fp16.h>
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

namespace arm_compute
{
class ITensor;

/** Interface for the activation layer kernel. */
class NEActivationLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEActivationLayerKernel";
    }
    /** Constructor */
    NEActivationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEActivationLayerKernel(const NEActivationLayerKernel &) = delete;
    /** Default move constructor */
    NEActivationLayerKernel(NEActivationLayerKernel &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEActivationLayerKernel &operator=(const NEActivationLayerKernel &) = delete;
    /** Default move assignment operator */
    NEActivationLayerKernel &operator=(NEActivationLayerKernel &&) = default;
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr, the activation function will be performed in-place
     *
     * @param[in, out] input           Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: QASYMM8/QSYMM16/F16/F32.
     * @param[out]     output          Destination tensor. Data type supported: same as @p input
     * @param[in]      activation_info Activation layer information.
     */
    void configure(ITensor *input, ITensor *output, ActivationLayerInfo activation_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEActivationLayerKernel
     *
     * @param[in] input    Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
     *                     of the activation function. Data types supported: QASYMM8/QSYMM16/F16/F32.
     * @param[in] output   Destination tensor info. Data type supported: same as @p input
     * @param[in] act_info Activation layer information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using ActivationFunction = ActivationLayerInfo::ActivationFunction;
    /** Common signature for all the specialised @ref NEActivationLayerKernel functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using ActivationFunctionExecutorPtr = void (NEActivationLayerKernel::*)(const Window &window);
    /** Function to apply an activation function on a tensor.
     *
     * @param[in] window Region on which to execute the kernel
     */
    template <ActivationLayerInfo::ActivationFunction F, typename T>
    typename std::enable_if<arm_compute::utils::traits::is_floating_point<T>::value, void>::type
    activation(const Window &window);
    /** Function to apply an activation function on a tensor.
     *
     * @param[in] window Region on which to execute the kernel
     */
    template <ActivationLayerInfo::ActivationFunction F, typename T>
    typename std::enable_if<std::is_same<T, qasymm8_t>::value, void>::type activation(const Window &window);
    /** Function to apply an activation function on a tensor.
     *
     * @param[in] window Region on which to execute the kernel
     */
    template <ActivationLayerInfo::ActivationFunction F, typename T>
    typename std::enable_if<std::is_same<T, qsymm16_t>::value, void>::type activation(const Window &window);

private:
    ITensor                      *_input;
    ITensor                      *_output;
    ActivationFunctionExecutorPtr _func;
    ActivationLayerInfo           _act_info;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEACTIVATIONLAYERKERNEL_H__ */
