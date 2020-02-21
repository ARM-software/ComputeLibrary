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
#ifndef ARM_COMPUTE_NEQUANTIZATIONLAYERKERNEL_H
#define ARM_COMPUTE_NEQUANTIZATIONLAYERKERNEL_H

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the quantization layer kernel.
 *
 * @note The implementation supports only 3D input tensors
 *
 */
class NEQuantizationLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEQuantizationLayerKernel";
    }
    /** Default constructor */
    NEQuantizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQuantizationLayerKernel(const NEQuantizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQuantizationLayerKernel &operator=(const NEQuantizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    NEQuantizationLayerKernel(NEQuantizationLayerKernel &&) = default;
    /** Default move assignment operator */
    NEQuantizationLayerKernel &operator=(NEQuantizationLayerKernel &&) = default;
    /** Default destructor */
    ~NEQuantizationLayerKernel() = default;
    /** Set the input, output.
     *
     * @param[in]  input  Source tensor. The dimensions over the third will be interpreted as batches. Data types supported: F32/F16.
     * @param[out] output Destination tensor with the same dimensions of input. Data types supported: QASYMM8/QASYMM16.
     *
     * @note Output auto initialization is not supported by this kernel
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEQuantizationLayerKernel
     *
     * @param[in] input  Input tensor info. Data types supported: F32/F16.
     * @param[in] output Output tensor info. Data types supported: QASYMM8/QASYMM16.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised @ref NEQuantizationLayerKernel functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using QuantizationFunctionExecutorPtr = void (NEQuantizationLayerKernel::*)(const Window &window);
    /** Function to apply QASYMM8 quantization on a tensor.
     *
     * @param[in] window Region on which to execute the kernel.
     */
    template <typename TIn, typename TOut>
    void run_quantize_qasymm8(const Window &window);
    /** Function to apply QASYMM16 quantization on a tensor.
     *
     * @param[in] window Region on which to execute the kernel.
     */
    template <typename T>
    void run_quantize_qasymm16(const Window &window);

    const ITensor *_input;
    ITensor       *_output;

    QuantizationFunctionExecutorPtr _func;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEQUANTIZATIONLAYERKERNEL_H */
