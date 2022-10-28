/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_H
#define ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/experimental/Types.h"
#include "src/cpu/ICpuKernel.h"
#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuDepthwiseConv2dNativeKernel.h"
#include "src/cpu/operators/CpuActivation.h"
#include "src/cpu/operators/CpuDepthwiseConv2dAssemblyDispatch.h"
#include "src/cpu/operators/CpuPermute.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
/** Function to execute a depthwise convolution.
 */
class CpuDepthwiseConv2d : public ICpuOperator
{
public:
    /** Default constructor */
    CpuDepthwiseConv2d() = default;
    /** Initialize the function's source, destination, weights and convolution information.
     *
     * @param[in, out] src     Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[out]     dst     Destination tensor info. Data type supported: same as @p src.
     * @param[in]      weights Weights tensor info. These are 3D tensor infos with shape [kernel_x, kernel_y, IFM].
     *                         Data type supported: Same as @p src or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p src is QASYMM8/QASYMM8_SIGNED.
     * @param[in]      biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                         Data type supported: Same as @p src, S32 when src is QASYMM8/QASYMM8_SIGNED.
     * @param[in]      info    Depthwise convolution meta-data.
     */
    void configure(ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ConvolutionInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDepthwiseConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info);
    /** Static function to choose the best depthwise convolution function for @ref CpuDepthwiseConv2d
     *
     * @param[in] src     Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in] weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                    Data type supported: Same as @p src or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p src is QASYMM8/QASYMM8_SIGNED.
     * @param[in] biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                    Data type supported: Same as @p src, S32 when src is QASYMM8/QASYMM8_SIGNED.
     * @param[in] dst     Destination tensor. Data type supported: same as @p src.
     * @param[in] info    Depthwise convolution meta-data.
     *
     * @return a Depthwise Convolution Function
     */
    static DepthwiseConvolutionFunction get_depthwiseconvolution_function(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                                                          const ConvolutionInfo &info);

    // Inherited methods overriden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &tensors) override;

private:
    /** Basic function to execute optimized depthwise convolution routines. This function calls the following kernels:
    *
    * @note At the moment 3x3 and 5x5 convolution of stride 1, 2 are supported
    *
    * -# @ref NEFillBorderKernel (if pad_x or pad_y > 0) and no assembly kernel implementation is present
    * -# @ref CpuDepthwiseConv2d3x3Kernel if 3x3 and no assembly kernel implementation is present
    * -# @ref CpuDepthwiseConv2dAssemblyDispatch if assembly kernel implementation is present
    * -# @ref CpuActivation if fused activation is required
    *
    */
    class CpuDepthwiseConv2dOptimizedInternal : public ICpuOperator
    {
    public:
        /** Default constructor */
        CpuDepthwiseConv2dOptimizedInternal() = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConv2dOptimizedInternal(const CpuDepthwiseConv2dOptimizedInternal &) = delete;
        /** Default move constructor */
        CpuDepthwiseConv2dOptimizedInternal(CpuDepthwiseConv2dOptimizedInternal &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConv2dOptimizedInternal &operator=(const CpuDepthwiseConv2dOptimizedInternal &) = delete;
        /** Default move assignment operator */
        CpuDepthwiseConv2dOptimizedInternal &operator=(CpuDepthwiseConv2dOptimizedInternal &&) = default;
        /** Default destructor */
        ~CpuDepthwiseConv2dOptimizedInternal() = default;
        /** Initialize the function's source, destination, kernels and border_size.
         *
         * @param[in, out] src     Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
         * @param[in]      weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p src.
         * @param[in]      biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                         Data type supported: Same as @p src, S32 when src is QASYMM8/QASYMM8_SIGNED.
         * @param[out]     dst     Destination tensor info. Data type supported: same as @p src.
         * @param[in]      info    Depthwise convolution meta-data.
         */
        void configure(ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ConvolutionInfo &info);
        /** Static function to check if given info will lead to a valid configuration
         *
         * Similar to CpuDepthwiseConv2dOptimizedInternal::configure()
         *
         * @return a status
         */
        static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info);

        // Inherited methods overriden:
        void run(ITensorPack &tensors) override;
        void prepare(ITensorPack &tensors) override;

    private:
        std::unique_ptr<CpuDepthwiseConv2dAssemblyDispatch> _dwc_optimized_func{ nullptr };
        std::unique_ptr<CpuPermute>                         _permute_input{ nullptr };
        std::unique_ptr<CpuPermute>                         _permute_weights{ nullptr };
        std::unique_ptr<CpuPermute>                         _permute_output{ nullptr };
        std::unique_ptr<CpuActivation>                      _activationlayer_function{ nullptr };
        bool                                                _has_bias{ false };
        bool                                                _is_quantized{ false };
        bool                                                _is_nchw{ true };
        bool                                                _permute{ false };
        bool                                                _is_activationlayer_enabled{ false };
        bool                                                _is_prepared{ false };
        bool                                                _are_weights_const{ true };
    };

    /** Basic function to execute a generic depthwise convolution. This function calls the following kernel:
     *
     * -# @ref CpuDepthwiseConv2dNativeKernel
     *
     */
    class CpuDepthwiseConv2dGeneric : public ICpuOperator
    {
    public:
        /** Default constructor */
        CpuDepthwiseConv2dGeneric() = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConv2dGeneric(const CpuDepthwiseConv2dGeneric &) = delete;
        /** Default move constructor */
        CpuDepthwiseConv2dGeneric(CpuDepthwiseConv2dGeneric &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConv2dGeneric &operator=(const CpuDepthwiseConv2dGeneric &) = delete;
        /** Default move assignment operator */
        CpuDepthwiseConv2dGeneric &operator=(CpuDepthwiseConv2dGeneric &&) = default;
        /** Default destructor */
        ~CpuDepthwiseConv2dGeneric() = default;
        /** Initialize the function's source, destination, weights and convolution information.
         *
         * @param[in, out] src     Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
         * @param[out]     dst     Destination tensor info. Data type supported: same as @p src.
         * @param[in]      weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                         Data type supported: Same as @p src or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p src is QASYMM8/QASYMM8_SIGNED.
         * @param[in]      biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                         Data type supported: Same as @p src, S32 when src is QASYMM8/QASYMM8_SIGNED.
         * @param[in]      info    Depthwise convolution meta-data.
         */
        void configure(ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ConvolutionInfo &info);

        /** Static function to check if given info will lead to a valid configuration
         *
         * Similar to CpuDepthwiseConv2dGeneric::configure()
         *
         * @return a status
         */
        static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info);

        // Inherited methods overridden:
        void run(ITensorPack &tensors) override;
        void prepare(ITensorPack &tensors) override;

    private:
        std::unique_ptr<kernels::CpuDepthwiseConv2dNativeKernel> _depthwise_conv_kernel{ nullptr };
        std::unique_ptr<CpuPermute>                              _permute_input{ nullptr };
        std::unique_ptr<CpuPermute>                              _permute_weights{ nullptr };
        std::unique_ptr<CpuPermute>                              _permute_output{ nullptr };
        std::unique_ptr<CpuActivation>                           _activationlayer_function{ nullptr };
        bool                                                     _is_nchw{ true };
        bool                                                     _is_prepared{ false };
        bool                                                     _is_activationlayer_enabled{ false };
    };

    DepthwiseConvolutionFunction        _depth_conv_func{ DepthwiseConvolutionFunction::GENERIC };
    CpuDepthwiseConv2dOptimizedInternal _func_optimized{};
    CpuDepthwiseConv2dGeneric           _func_generic{};
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_H */
