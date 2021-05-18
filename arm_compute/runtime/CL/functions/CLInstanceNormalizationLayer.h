/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLINSTANCENORMALIZATIONLAYER_H
#define ARM_COMPUTE_CLINSTANCENORMALIZATIONLAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;
class ICLKernel;
class CLRuntimeContext;

/** Basic function to perform a Instance normalization.
 *
 * This function runs the following kernels:
 * -# @ref CLInstanceNormalizationLayerKernel
 */
class CLInstanceNormalizationLayer : public IFunction
{
public:
    /** Constructor
     *
     * @param[in] ctx Runtime context to be used by the function
     */
    CLInstanceNormalizationLayer(CLRuntimeContext *ctx = nullptr);

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLInstanceNormalizationLayer(const CLInstanceNormalizationLayer &) = delete;
    /** Default move constructor */
    CLInstanceNormalizationLayer(CLInstanceNormalizationLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLInstanceNormalizationLayer &operator=(const CLInstanceNormalizationLayer &) = delete;
    /** Default move assignment operator */
    CLInstanceNormalizationLayer &operator=(CLInstanceNormalizationLayer &&) = default;
    /** Default destructor */
    ~CLInstanceNormalizationLayer();

    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src      |dst       |
     * |:--------|:---------|
     * |F16      |F16       |
     * |F32      |F32       |
     *
     * @param[in, out] input               Source tensor. In case of @p output tensor = nullptr this tensor will store the result of the normalization.
     *                                     Data types supported: F16/F32. Data layout supported: NHWC, NCHW
     * @param[out]     output              Destination tensor. Data types and data layouts supported: same as @p input.
     * @param[in]      gamma               (Optional) The scale scalar value applied to the normalized tensor. Defaults to 1.0
     * @param[in]      beta                (Optional) The offset scalar value applied to the normalized tensor. Defaults to 0.0
     * @param[in]      epsilon             (Optional) Lower bound value for the normalization. Defaults to 1e-12
     * @param[in]      use_mixed_precision (Optional) Use mixed precision in case of FP16 execution
     */
    void configure(ICLTensor *input, ICLTensor *output, float gamma = 1.0f, float beta = 0.0f, float epsilon = 1e-12f, bool use_mixed_precision = true);
    /** Set the input and output tensors.
     *
     * @param[in]      compile_context     The compile context to be used.
     * @param[in, out] input               Source tensor. In case of @p output tensor = nullptr this tensor will store the result of the normalization.
     *                                     Data types supported: F16/F32. Data layout supported: NHWC, NCHW
     * @param[out]     output              Destination tensor. Data types and data layouts supported: same as @p input.
     * @param[in]      gamma               (Optional) The scale scalar value applied to the normalized tensor. Defaults to 1.0
     * @param[in]      beta                (Optional) The offset scalar value applied to the normalized tensor. Defaults to 0.0
     * @param[in]      epsilon             (Optional) Lower bound value for the normalization. Defaults to 1e-12
     * @param[in]      use_mixed_precision (Optional) Use mixed precision in case of FP16 execution
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, float gamma = 1.0f, float beta = 0.0f, float epsilon = 1e-12f, bool use_mixed_precision = true);

    /** Static function to check if given info will lead to a valid configuration of @ref CLInstanceNormalizationLayer.
     *
     * @param[in] input               Source tensor info. Data types supported: F16/F32. Data layout supported: NHWC, NCHW
     * @param[in] output              Destination tensor info. Data types and data layouts supported: same as @p input.
     * @param[in] gamma               (Optional) The scale scalar value applied to the normalized tensor. Defaults to 1.0
     * @param[in] beta                (Optional) The offset scalar value applied to the normalized tensor. Defaults to 0.0
     * @param[in] epsilon             (Optional) Lower bound value for the normalization. Defaults to 1e-12
     * @param[in] use_mixed_precision (Optional) Use mixed precision in case of FP16 execution
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, float gamma = 1.0f, float beta = 0.0f, float epsilon = 1e-12f, bool use_mixed_precision = true);
    void run() override;

private:
    std::unique_ptr<ICLKernel> _inst_norm_kernel; /**< Kernel to run */
    std::unique_ptr<ICLKernel> _mean_var_kernel;  /**< Kernel to run */
    CLTensor                   _mean_var_tensor;
    CLRuntimeContext          *_ctx; /**< Context to use */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLINSTANCENORMALIZATIONLAYER_H */
