/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEFUSEBATCHNORMALIZATION_H__
#define __ARM_COMPUTE_NEFUSEBATCHNORMALIZATION_H__

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEFuseBatchNormalizationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to fuse the batch normalization node to a preceding convolution node */
class NEFuseBatchNormalization : public IFunction
{
public:
    /** Default constructor */
    NEFuseBatchNormalization();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFuseBatchNormalization(const NEFuseBatchNormalization &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFuseBatchNormalization &operator=(const NEFuseBatchNormalization &) = delete;
    /** Allow instances of this class to be moved */
    NEFuseBatchNormalization(NEFuseBatchNormalization &&) = default;
    /** Allow instances of this class to be moved */
    NEFuseBatchNormalization &operator=(NEFuseBatchNormalization &&) = default;
    /** Default destructor */
    ~NEFuseBatchNormalization() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  conv_weights  Convolution layer weights tensor. Data type supported: F16/F32
     * @param[in]  bn_mean       Batch normalization layer mean tensor. Same as @p conv_weights
     * @param[in]  bn_var        Batch normalization layer variance tensor. Same as @p conv_weights
     * @param[out] fused_weights Output fused weights tensor. Same as @p conv_weights
     * @param[out] fused_bias    Output fused bias tensor. Same as @p conv_weights
     * @param[in]  conv_bias     (Optional) Convolution layer bias tensor. Same as @p conv_weights
     * @param[in]  bn_beta       (Optional) Batch normalization layer beta tensor. Same as @p conv_weights
     * @param[in]  bn_gamma      (Optional) Batch normalization layer gamma tensor. Same as @p conv_weights
     * @param[in]  epsilon       (Optional) Batch normalization layer epsilon parameter. Defaults to 0.001f.
     */
    void configure(const ITensor *conv_weights, const ITensor *bn_mean, const ITensor *bn_var, ITensor *fused_weights, ITensor *fused_bias,
                   const ITensor *conv_bias = nullptr, const ITensor *bn_beta = nullptr, const ITensor *bn_gamma = nullptr,
                   float epsilon = 0.001f);
    /** Static function to check if given info will lead to a valid configuration of @ref NEFuseBatchNormalization
     *
     * @param[in] conv_weights  Convolution layer weights tensor. Data type supported: F16/F32
     * @param[in] bn_mean       Batch normalization layer mean tensor. Same as @p conv_weights
     * @param[in] bn_var        Batch normalization layer variance tensor. Same as @p conv_weights
     * @param[in] fused_weights Output fused weights tensor. Same as @p conv_weights
     * @param[in] fused_bias    Output fused bias tensor. Same as @p conv_weights
     * @param[in] conv_bias     (Optional) Convolution layer bias tensor. Same as @p conv_weights
     * @param[in] bn_beta       (Optional) Batch normalization layer beta tensor. Same as @p conv_weights
     * @param[in] bn_gamma      (Optional) Batch normalization layer gamma tensor. Same as @p conv_weights
     * @param[in] epsilon       (Optional) Batch normalization layer epsilon parameter. Defaults to 0.001f.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *conv_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                           const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                           const ITensorInfo *conv_bias = nullptr, const ITensorInfo *bn_beta = nullptr, const ITensorInfo *bn_gamma = nullptr,
                           float epsilon = 0.001f);

    // Inherited methods overridden:
    void run() override;

private:
    NEFuseBatchNormalizationKernel _fuse_bn_kernel;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEFUSEBATCHNORMALIZATION_H__ */
