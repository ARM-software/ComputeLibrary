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
#ifndef ARM_COMPUTE_NEBATCHNORMALIZATIONLAYERKERNEL_H
#define ARM_COMPUTE_NEBATCHNORMALIZATIONLAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the batch normalization layer kernel.
 */
class NEBatchNormalizationLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEBatchNormalizationLayerKernel";
    }
    /** Default constructor */
    NEBatchNormalizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchNormalizationLayerKernel(const NEBatchNormalizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchNormalizationLayerKernel &operator=(const NEBatchNormalizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    NEBatchNormalizationLayerKernel(NEBatchNormalizationLayerKernel &&) = default;
    /** Default move assignment operator */
    NEBatchNormalizationLayerKernel &operator=(NEBatchNormalizationLayerKernel &&) = default;
    /** Default destructor */
    ~NEBatchNormalizationLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @note If the output tensor is a nullptr, the batch normalization function will be performed in-place
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
    /** Static function to check if given info will lead to a valid configuration of @ref NEBatchNormalizationLayerKernel
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
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Configure execution function in case of non-fused activation **/
    void configure_non_fused();
    /** Configure execution function in case of fused activation **/
    void configure_fused();

    /** Template function to run batch normalization on fp32
     *
     * @tparam T                Specialization data type
     * @tparam fused_activation Boolean that flags if its a fused activation or not
     * @tparam F                Activation function functor to run
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T, bool fused_activation, typename F>
    void batch_normalization_nchw(const Window &window);
    /** Template function to run batch normalization on fp32 on tensors with NHWC format
     *
     * @tparam T                Specialization data type
     * @tparam fused_activation Boolean that flags if its a fused activation or not
     * @tparam F                Activation function functor to run
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T, bool fused_activation, typename F>
    void batch_normalization_nhwc(const Window &window);
    /** Common signature for all the batch normalization functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using BatchNormFunctionPtr = void (NEBatchNormalizationLayerKernel::*)(const Window &window);

private:
    BatchNormFunctionPtr _func;
    ITensor             *_input;
    ITensor             *_output;
    const ITensor       *_mean;
    const ITensor       *_var;
    const ITensor       *_gamma;
    const ITensor       *_beta;
    float                _epsilon;
    ActivationLayerInfo  _act_info;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEBATCHNORMALIZATIONLAYERKERNEL_H */
