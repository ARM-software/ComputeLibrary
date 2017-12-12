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
#ifndef __ARM_COMPUTE__H__
#define __ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYERKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the NormalizePlanarYUV layer kernel.
 */
class GCNormalizePlanarYUVLayerKernel : public IGCKernel
{
public:
    /** Constructor */
    GCNormalizePlanarYUVLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCNormalizePlanarYUVLayerKernel(const GCNormalizePlanarYUVLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCNormalizePlanarYUVLayerKernel &operator=(const GCNormalizePlanarYUVLayerKernel &) = delete;
    /** Default Move Constructor. */
    GCNormalizePlanarYUVLayerKernel(GCNormalizePlanarYUVLayerKernel &&) = default;
    /** Default move assignment operator. */
    GCNormalizePlanarYUVLayerKernel &operator=(GCNormalizePlanarYUVLayerKernel &&) = default;
    /** Default destructor */
    ~GCNormalizePlanarYUVLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. 3 lower dimensions represent a single input with dimensions [width, height, FM].
     *                    Data types supported: F16.
     * @param[out] output Destination tensor. Output will have the same number of dimensions as input. Data type supported: same as @p input
     * @param[in]  mean   Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  sd     Standard deviation values tensor. 1 dimension with size equal to the feature maps [FM].
     *                     Data types supported: Same as @p input
     */
    void configure(const IGCTensor *input, IGCTensor *output, const IGCTensor *mean, const IGCTensor *sd);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    IGCTensor       *_output;
    const IGCTensor *_mean;
    const IGCTensor *_sd;
};
}
#endif /*__ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYERKERNEL_H__ */
