/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLFUSEBATCHNORMALIZATION_H
#define ARM_COMPUTE_CLFUSEBATCHNORMALIZATION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class CLFuseBatchNormalizationKernel;
class ICLTensor;
class ITensorInfo;

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
    ~CLFuseBatchNormalization();
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
     * @param[in]  input_weights Input weights tensor for convolution or depthwise convolution layer. Data type supported: F16/F32. Data layout supported: NCHW, NHWC
     * @param[in]  bn_mean       Batch normalization layer mean tensor. Same as @p input_weights
     * @param[in]  bn_var        Batch normalization layer variance tensor. Same as @p input_weights
     * @param[out] fused_weights Output fused weights tensor. It can be a nullptr in case of in-place computation. Same as @p input_weights
     * @param[out] fused_bias    Output fused bias tensor. It can be a nullptr in case of in-place computation and input_bias != nullptr. Same as @p input_weights
     * @param[in]  input_bias    (Optional) Input bias tensor for convolution or depthwise convolution layer. It can be a nullptr in case the bias tensor is not required. Same as @p input_weights
     * @param[in]  bn_beta       (Optional) Batch normalization layer beta tensor. It can be a nullptr in case the beta tensor is not required. Same as @p input_weights
     *                           @note if nullptr, bn_beta is set to 0.0
     * @param[in]  bn_gamma      (Optional) Batch normalization layer gamma tensor. It can be a nullptr in case the gamma tensor is not required. Same as @p input_weights
     *                           @note if nullptr, bn_gamma is set to 1.0
     * @param[in]  epsilon       (Optional) Batch normalization layer epsilon parameter. Defaults to 0.001f.
     * @param[in]  fbn_type      (Optional) Fused batch normalization type. Defaults to Convolution.
     */
    void configure(const ICLTensor *input_weights, const ICLTensor *bn_mean, const ICLTensor *bn_var, ICLTensor *fused_weights, ICLTensor *fused_bias,
                   const ICLTensor *input_bias = nullptr, const ICLTensor *bn_beta = nullptr, const ICLTensor *bn_gamma = nullptr,
                   float epsilon = 0.001f, FuseBatchNormalizationType fbn_type = FuseBatchNormalizationType::CONVOLUTION);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input_weights   Input weights tensor for convolution or depthwise convolution layer. Data type supported: F16/F32. Data layout supported: NCHW, NHWC
     * @param[in]  bn_mean         Batch normalization layer mean tensor. Same as @p input_weights
     * @param[in]  bn_var          Batch normalization layer variance tensor. Same as @p input_weights
     * @param[out] fused_weights   Output fused weights tensor. It can be a nullptr in case of in-place computation. Same as @p input_weights
     * @param[out] fused_bias      Output fused bias tensor. It can be a nullptr in case of in-place computation and input_bias != nullptr. Same as @p input_weights
     * @param[in]  input_bias      (Optional) Input bias tensor for convolution or depthwise convolution layer. It can be a nullptr in case the bias tensor is not required. Same as @p input_weights
     * @param[in]  bn_beta         (Optional) Batch normalization layer beta tensor. It can be a nullptr in case the beta tensor is not required. Same as @p input_weights
     *                             @note if nullptr, bn_beta is set to 0.0
     * @param[in]  bn_gamma        (Optional) Batch normalization layer gamma tensor. It can be a nullptr in case the gamma tensor is not required. Same as @p input_weights
     *                             @note if nullptr, bn_gamma is set to 1.0
     * @param[in]  epsilon         (Optional) Batch normalization layer epsilon parameter. Defaults to 0.001f.
     * @param[in]  fbn_type        (Optional) Fused batch normalization type. Defaults to Convolution.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input_weights, const ICLTensor *bn_mean, const ICLTensor *bn_var, ICLTensor *fused_weights, ICLTensor *fused_bias,
                   const ICLTensor *input_bias = nullptr, const ICLTensor *bn_beta = nullptr, const ICLTensor *bn_gamma = nullptr,
                   float epsilon = 0.001f, FuseBatchNormalizationType fbn_type = FuseBatchNormalizationType::CONVOLUTION);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFuseBatchNormalization
     *
     * @param[in] input_weights Input weights tensor info for convolution or depthwise convolution layer. Data type supported: F16/F32. Data layout supported: NCHW, NHWC
     * @param[in] bn_mean       Batch normalization layer mean tensor info. Same as @p input_weights
     * @param[in] bn_var        Batch normalization layer variance tensor info. Same as @p input_weights
     * @param[in] fused_weights Output fused weights tensor info. It can be a nullptr in case of in-place computation. Same as @p input_weights
     * @param[in] fused_bias    Output fused bias tensor info. It can be a nullptr in case of in-place computation and input_bias != nullptr. Same as @p input_weights
     * @param[in] input_bias    (Optional) Input bias tensor info for convolution or depthwise convolution layer. It can be a nullptr in case the bias tensor is not required. Same as @p input_weights
     * @param[in] bn_beta       (Optional) Batch normalization layer beta tensor info. It can be a nullptr in case the beta tensor is not required. Same as @p input_weights
     *                          @note if nullptr, bn_beta is set to 0.0
     * @param[in] bn_gamma      (Optional) Batch normalization layer gamma tensor info. It can be a nullptr in case the gamma tensor is not required. Same as @p input_weights
     *                          @note if nullptr, bn_gamma is set to 1.0
     * @param[in] epsilon       (Optional) Batch normalization layer epsilon parameter. Defaults to 0.001f.
     * @param[in] fbn_type      (Optional) Fused batch normalization type. Defaults to Convolution.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                           const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                           const ITensorInfo *input_bias = nullptr, const ITensorInfo *bn_beta = nullptr, const ITensorInfo *bn_gamma = nullptr,
                           float epsilon = 0.001f, FuseBatchNormalizationType fbn_type = FuseBatchNormalizationType::CONVOLUTION);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<CLFuseBatchNormalizationKernel> _fuse_bn_kernel;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLFUSEBATCHNORMALIZATION_H */
