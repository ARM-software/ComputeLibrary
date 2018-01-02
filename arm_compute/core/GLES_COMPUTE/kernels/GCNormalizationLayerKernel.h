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
#ifndef __ARM_COMPUTE_GCNORMALIZATIONLAYERKERNEL_H__
#define __ARM_COMPUTE_GCNORMALIZATIONLAYERKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the normalization layer kernel.
 */
class GCNormalizationLayerKernel : public IGCKernel
{
public:
    /** Constructor */
    GCNormalizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCNormalizationLayerKernel(const GCNormalizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCNormalizationLayerKernel &operator=(const GCNormalizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    GCNormalizationLayerKernel(GCNormalizationLayerKernel &&) = default;
    /** Default move assignment operator. */
    GCNormalizationLayerKernel &operator=(GCNormalizationLayerKernel &&) = default;
    /** Default destrutor */
    ~GCNormalizationLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input         Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                           and an optional 4th dimension for batch of inputs. Data types supported: F32.
     * @param[in]  squared_input Source with each element has been squared. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                           Data types should match the input type.
     * @param[out] output        Destination tensor. Output will have the same number of dimensions as input. Data types should match the input type.
     * @param[in]  norm_info     Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(const IGCTensor *input, const IGCTensor *squared_input, IGCTensor *output, NormalizationLayerInfo norm_info);

    // Inherited methods overridden:
    void run(const Window &window) override;
    BorderSize border_size() const override;

private:
    const IGCTensor *_input;
    const IGCTensor *_squared_input;
    IGCTensor       *_output;
    BorderSize       _border_size;
};
}
#endif /*__ARM_COMPUTE_GCNORMALIZATIONLAYERKERNEL_H__ */
