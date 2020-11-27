/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYER_H
#define ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizePlanarYUVLayerKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to run @ref GCNormalizePlanarYUVLayerKernel
 *
 *  @note The function simulates a NormalizePlanarYUV layer.
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCNormalizePlanarYUVLayer : public IFunction
{
public:
    /** Default constructor */
    GCNormalizePlanarYUVLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. 3 lower dimensions represent a single input with dimensions [width, height, channels].
     *                    Data types supported: F16.
     * @param[out] output Destination tensor. Data type supported: same as @p input
     * @param[in]  mean   Mean values tensor. 1 dimension with size equal to the number of input channels. Data types supported: same as @p input
     * @param[in]  std    Standard deviation values tensor. 1 dimension with size equal to the number of input channels.
     *                    Data types supported: same as @p input
     */
    void configure(const IGCTensor *input, IGCTensor *output, const IGCTensor *mean, const IGCTensor *std);
    /** Static function to check if given info will lead to a valid configuration of @ref CLNormalizePlanarYUVLayer
     *
     * @param[in]  input  Source tensor info. 3 lower dimensions represent a single input with dimensions [width, height, channels].
     *                    Data types supported: F16/F32.
     * @param[out] output Destination tensor info. Data type supported: same as @p input
     * @param[in]  mean   Mean values tensor info. 1 dimension with size equal to the number of input channels. Data types supported: same as @p input
     * @param[in]  std    Standard deviation values tensor info. 1 dimension with size equal to the number of input channels.
     *                    Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *std);

    // Inherited methods overridden:
    void run() override;

private:
    GCNormalizePlanarYUVLayerKernel _norm_kernel; /**< NormalizePlanarYUV layer kernel to run */
};
}
#endif /* ARM_COMPUTE_GCNORMALIZEPLANARYUVLAYER_H */
