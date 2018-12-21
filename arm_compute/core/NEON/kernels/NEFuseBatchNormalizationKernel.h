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
#ifndef __ARM_COMPUTE_NEFUSEBATCHNORMALIZATIONKERNEL_H__
#define __ARM_COMPUTE_NEFUSEBATCHNORMALIZATIONKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** OpenNE kernel to fuse the batch normalization node to a preceding convolution node */
class NEFuseBatchNormalizationKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEFuseBatchNormalizationKernel";
    }
    /** Default constructor */
    NEFuseBatchNormalizationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFuseBatchNormalizationKernel(const NEFuseBatchNormalizationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFuseBatchNormalizationKernel &operator=(const NEFuseBatchNormalizationKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEFuseBatchNormalizationKernel(NEFuseBatchNormalizationKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEFuseBatchNormalizationKernel &operator=(NEFuseBatchNormalizationKernel &&) = default;
    /** Default destructor */
    ~NEFuseBatchNormalizationKernel() = default;
    /** Set the source, destination of the kernel
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
    /** Static function to check if given info will lead to a valid configuration of @ref NEFuseBatchNormalizationKernel
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
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_conv_weights;
    const ITensor *_conv_bias;
    const ITensor *_bn_mean;
    const ITensor *_bn_var;
    const ITensor *_bn_gamma;
    const ITensor *_bn_beta;
    ITensor       *_fused_weights;
    ITensor       *_fused_bias;
    float          _epsilon;
    bool           _run_in_place_weights;
    bool           _run_in_place_bias;

    using FuseBatchNormFunction = void(const ITensor *conv_weights, const ITensor *conv_bias, ITensor *fused_weights, ITensor *fused_bias,
                                       const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window);

    FuseBatchNormFunction *_func;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEFUSEBATCHNORMALIZATIONKERNEL_H__ */
