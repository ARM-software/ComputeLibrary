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
#ifndef __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERBIASACCUMULATEKERNEL_H__
#define __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERBIASACCUMULATEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;
/** NEON kernel to accumulate the biases to each element of the input tensor
 *
 * @note We assume bias to be shared
 */
class NEDirectConvolutionLayerBiasAccumulateKernel : public INEKernel
{
public:
    /** Default constructor */
    NEDirectConvolutionLayerBiasAccumulateKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerBiasAccumulateKernel(const NEDirectConvolutionLayerBiasAccumulateKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerBiasAccumulateKernel &operator=(const NEDirectConvolutionLayerBiasAccumulateKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerBiasAccumulateKernel(NEDirectConvolutionLayerBiasAccumulateKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerBiasAccumulateKernel &operator=(NEDirectConvolutionLayerBiasAccumulateKernel &&) = default;
    /** Default destructor */
    ~NEDirectConvolutionLayerBiasAccumulateKernel() = default;
    /** Set the accumulate buffer and the biases of the kernel.
     *
     * @param[in, out] input  Input to add the bias to. If @p output is not specified then accumulation is done in-place.
     *                        Data type supported: QS8/QS16/F16/F32
     * @param[in]      bias   The shared bias tensor to add. It must be 1D Tensor. Data type supported: Same as @p input
     * @param[out]     output (Optional) If the output tensor is specified the accumulation is done out-of-place. (Defaults to nullptr)
     *                         Data type supported: Same as @p input
     */
    void configure(ITensor *input, const ITensor *bias, ITensor *output = nullptr);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    using BiasAccumulateKernel = void(ITensor *input, const ITensor *bias, const Window window, ITensor *output);

private:
    BiasAccumulateKernel *_func;
    ITensor              *_input;
    const ITensor        *_bias;
    ITensor              *_output;
};
}
#endif /*__ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERBIASACCUMULATEKERNEL_H__ */
