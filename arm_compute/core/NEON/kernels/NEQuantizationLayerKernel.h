/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEQUANTIZATIONLAYERKERNEL_H__
#define __ARM_COMPUTE_NEQUANTIZATIONLAYERKERNEL_H__

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
    /** Default constructor */
    NEQuantizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQuantizationLayerKernel(const NEQuantizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQuantizationLayerKernel &operator=(const NEQuantizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    NEQuantizationLayerKernel(NEQuantizationLayerKernel &&) = default;
    /** Default move assignment operator. */
    NEQuantizationLayerKernel &operator=(NEQuantizationLayerKernel &&) = default;
    /** Default destructor */
    ~NEQuantizationLayerKernel() = default;
    /** Set the input, output, min and max.
     *
     * @param[in]  input   Source tensor with at least 3 dimensions. The dimensions over the third will be interpreted as batches. Data types supported: F32.
     * @param[out] output  Destination tensor with the same dimensions of input. Data types supported: U8.
     * @param[in]  min_max Pointer to the tensor with shape [2, batches] which stores the minimum and maximum value for each 3D input tensor.
     *                     The dimensions over the second must match the batched dimensions of the input tensor. Data type supported: F32
     */
    void configure(const ITensor *input, ITensor *output, const ITensor *min_max);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
    const ITensor *_min_max;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEQUANTIZATIONLAYERKERNEL_H__ */
