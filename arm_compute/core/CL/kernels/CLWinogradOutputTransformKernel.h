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
#ifndef __ARM_COMPUTE_CLWINOGRADOUTPUTTRANSFORMKERNEL_H__
#define __ARM_COMPUTE_CLWINOGRADOUTPUTTRANSFORMKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the Winograd output transform kernel. */
class CLWinogradOutputTransformKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLWinogradOutputTransformKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradOutputTransformKernel(const CLWinogradOutputTransformKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLWinogradOutputTransformKernel &operator=(const CLWinogradOutputTransformKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLWinogradOutputTransformKernel(CLWinogradOutputTransformKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLWinogradOutputTransformKernel &operator=(CLWinogradOutputTransformKernel &&) = default;
    /** Default destructor */
    ~CLWinogradOutputTransformKernel() = default;
    /** Set the input and output tensor.
     *
     * @note Winograd output transform supports the following configurations for NCWH data layout
     *       F(output tile, kernel size):F(2x2, 3x3), F(2x1, 3x1), F(1x2, 1x3),
     *                                   F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     * @note Winograd output transform supports the following configurations for NHWC data layout
     *       F(output tile, kernel size):F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     *       Strides: only unit strides
     *
     * @param[in]  input         Source tensor with shape [C, N, K, batches]. Data types supported: F16/F32.
     * @param[in]  bias          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. It can be a nullptr. Data type supported: as @p input
     * @param[out] output        The output tensor. The shape for this tensor can be calculated using the utility function @p compute_winograd_output_transform_shape. Data types supported: Same as @p input
     * @param[in]  winograd_info Contains Winograd's information described in @ref WinogradInfo
     * @param[in]  act_info      (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    /** Static function to check if given info will lead to a valid configuration of @ref CLWinogradOutputTransformKernel
     *
     * @note Winograd output transform supports the following configurations for NCWH data layout
     *       F(output tile, kernel size):F(2x2, 3x3), F(2x1, 3x1), F(1x2, 1x3),
     *                                   F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     * @note Winograd output transform supports the following configurations for NHWC data layout
     *       F(output tile, kernel size):F(4x4, 3x3), F(4x1, 3x1), F(1x4, 1x3),
     *                                   F(4x4, 5x5), F(4x1, 5x1), F(1x4, 1x5)
     *
     *       Strides: only unit strides
     *
     * @param[in]  input         Source tensor with shape [C, N, K, batches]. Data types supported: F16/F32.
     * @param[in]  bias          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. It can be a nullptr. Data type supported: as @p input
     * @param[out] output        The output tensor. The shape for this tensor can be calculated using the utility function @p compute_winograd_output_transform_shape. Data types supported: Same as @p input
     * @param[in]  winograd_info Contains Winograd's information described in @ref WinogradInfo
     * @param[in]  act_info      (Optional) Activation layer information in case of a fused activation @ref ActivationLayerInfo. Only RELU, BOUNDED_RELU, LU_BOUNDED_RELU, LEAKY_RELU and SOFT_RELU supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    using WinogradKey = std::pair<std::pair<int, int>, std::pair<int, int>>;

    const ICLTensor *_input;
    const ICLTensor *_bias;
    ICLTensor       *_output;
    bool             _is_nhwc;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLWINOGRADOUTPUTTRANSFORMKERNEL_H__ */
