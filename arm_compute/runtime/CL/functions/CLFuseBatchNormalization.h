/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLFUSEBATCHNORMALIZATION_H__
#define __ARM_COMPUTE_CLFUSEBATCHNORMALIZATION_H__

#include "arm_compute/core/CL/kernels/CLFuseBatchNormalizationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to fuse the batch normalization node to a preceding convolution node */
class CLFuseBatchNormalization : public IFunction
{
public:
    /** Default constructor */
    CLFuseBatchNormalization();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFuseBatchNormalization(const CLFuseBatchNormalization &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFuseBatchNormalization &operator=(const CLFuseBatchNormalization &) = delete;
    /** Allow instances of this class to be moved */
    CLFuseBatchNormalization(CLFuseBatchNormalization &&) = default;
    /** Allow instances of this class to be moved */
    CLFuseBatchNormalization &operator=(CLFuseBatchNormalization &&) = default;
    /** Default destructor */
    ~CLFuseBatchNormalization() = default;
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
    void configure(const ICLTensor *conv_weights, const ICLTensor *bn_mean, const ICLTensor *bn_var, ICLTensor *fused_weights, ICLTensor *fused_bias,
                   const ICLTensor *conv_bias = nullptr, const ICLTensor *bn_beta = nullptr, const ICLTensor *bn_gamma = nullptr,
                   float epsilon = 0.001f);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFuseBatchNormalization
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
    CLFuseBatchNormalizationKernel _fuse_bn_kernel;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLFUSEBATCHNORMALIZATION_H__ */
