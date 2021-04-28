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
#ifndef ARM_COMPUTE_NEBATCHNORMALIZATIONLAYER_H
#define ARM_COMPUTE_NEBATCHNORMALIZATIONLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class NEBatchNormalizationLayerKernel;

/** Basic function to run @ref NENormalizationLayerKernel and simulate a batch normalization layer.
 *
 * Batch normalization is calculated by:
 * @f[ out_i = \gamma * (\frac{in_i - \mu_{B}}{\sqrt{\sigma^2_{B} + \epsilon}}) + \beta \equiv BN_{\gamma,\beta}(in_i) @f]
 *
 */
class NEBatchNormalizationLayer : public IFunction
{
public:
    /** Constructor */
    NEBatchNormalizationLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchNormalizationLayer(const NEBatchNormalizationLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchNormalizationLayer &operator=(const NEBatchNormalizationLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEBatchNormalizationLayer(NEBatchNormalizationLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEBatchNormalizationLayer &operator=(NEBatchNormalizationLayer &&) = delete;
    /** Default destructor */
    ~NEBatchNormalizationLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F32            |F32            |
     * |F16            |F16            |
     *
     * @note If the output tensor is a nullptr or is equal to the input, the batch normalization function will be performed in-place
     *
     * @param[in, out] input    Source tensor. In case of @p output tensor = nullptr, this tensor will store the result.
     *                          3 lower dimensions represent a single input with dimensions [width, height, FM].
     *                          The rest are optional and used for representing batches. Data types supported: F16/F32.
     * @param[out]     output   Destination tensor. Output will have the same number of dimensions as input. Data type supported: same as @p input
     * @param[in]      mean     Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]      var      Variance values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]      beta     (Optional) Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
     * @param[in]      gamma    (Optional) Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
     * @param[in]      epsilon  (Optional) Small value to avoid division with zero. Default value is 0.001f.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     */
    void configure(ITensor *input, ITensor *output, const ITensor *mean, const ITensor *var, const ITensor *beta = nullptr, const ITensor *gamma = nullptr, float epsilon = 0.001f,
                   ActivationLayerInfo act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEBatchNormalizationLayer
     *
     * @param[in] input    Source tensor info. In case of @p output tensor = nullptr, this tensor will store the result.
     *                     3 lower dimensions represent a single input with dimensions [width, height, FM].
     *                     The rest are optional and used for representing batches. Data types supported: F16/F32.
     * @param[in] output   Destination tensor info. Output will have the same number of dimensions as input. Data type supported: same as @p input
     * @param[in] mean     Mean values tensor info. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in] var      Variance values tensor info. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in] beta     (Optional) Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
     * @param[in] gamma    (Optional) Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
     * @param[in] epsilon  (Optional) Small value to avoid division with zero. Default value is 0.001f.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                           const ITensorInfo *mean, const ITensorInfo *var,
                           const ITensorInfo *beta = nullptr, const ITensorInfo *gamma = nullptr,
                           float epsilon = 0.001f, ActivationLayerInfo act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEBatchNormalizationLayerKernel> _norm_kernel; /**< Batch normalization layer kernel */
};
}
#endif /* ARM_COMPUTE_NEBATCHNORMALIZATIONLAYER_H */
