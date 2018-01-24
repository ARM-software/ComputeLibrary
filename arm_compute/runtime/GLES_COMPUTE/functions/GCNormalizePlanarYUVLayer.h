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
#ifndef __ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYER_H__
#define __ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizePlanarYUVLayerKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to run @ref GCNormalizePlanarYUVLayerKernel
 *
 *  @note The function simulates a NormalizePlanarYUV layer.
 */
class GCNormalizePlanarYUVLayer : public IFunction
{
public:
    /** Default constructor */
    GCNormalizePlanarYUVLayer();
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
    void run() override;

private:
    GCNormalizePlanarYUVLayerKernel _norm_kernel; /**< NormalizePlanarYUV layer kernel to run */
};
}
#endif /* __ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYER_H__ */
