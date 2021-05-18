/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_DEQUANTIZATION_H
#define ARM_COMPUTE_CPU_DEQUANTIZATION_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/experimental/Types.h"
#include "src/core/cpu/ICpuKernel.h"
#include "src/core/cpu/kernels/CpuDepthwiseConvolutionNativeKernel.h"
#include "src/runtime/cpu/ICpuOperator.h"
#include "src/runtime/cpu/operators/CpuActivation.h"
#include "src/runtime/cpu/operators/CpuDepthwiseConvolutionAssemblyDispatch.h"
#include "src/runtime/cpu/operators/CpuPermute.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
/** Function to execute a depthwise convolution.
 */
class CpuDepthwiseConvolution : public ICpuOperator
{
public:
    /** Default constructor */
    CpuDepthwiseConvolution();
    /** Initialize the function's source, destination, weights and convolution information.
     *
     * @param[in, out] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[out]     output  Destination tensor info. Data type supported: same as @p input.
     * @param[in]      weights Weights tensor info. These are 3D tensor infos with shape [kernel_x, kernel_y, IFM].
     *                         Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]      biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                         Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]      info    Depthwise convolution meta-data.
     */
    void configure(ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *output, const ConvolutionInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CpuDepthwiseConvolution
     *
     * @param[in] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in] output  Destination tensor info. Data type supported: same as @p input.
     * @param[in] weights Weights tensor info. These are 3D tensors info with shape [kernel_x, kernel_y, IFM].
     *                    Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                    Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] info    Depthwise convolution meta-data.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const ConvolutionInfo &info);

    /** Static function to choose the best depthwise convolution function for @ref CpuDepthwiseConvolution
     *
     * @param[in] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in] weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                    Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                    Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] output  Destination tensor. Data type supported: same as @p input.
     * @param[in] info    Depthwise convolution meta-data.
     *
     * @return a Depthwise Convolution Function
     */
    static DepthwiseConvolutionFunction get_depthwiseconvolution_function(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
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
    * -# @ref CpuDepthwiseConvolution3x3Kernel if 3x3 and no assembly kernel implementation is present
    * -# @ref NEDepthwiseConvolutionAssemblyDispatch if assembly kernel implementation is present
    * -# @ref NEDirectConvolutionLayerOutputStageKernel if re-quantization of output is required
    * -# @ref NEActivationLayer if fused activation is required
    *
    */
    class CpuDepthwiseConvolutionOptimizedInternal : public ICpuOperator
    {
    public:
        /** Default constructor */
        CpuDepthwiseConvolutionOptimizedInternal();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConvolutionOptimizedInternal(const CpuDepthwiseConvolutionOptimizedInternal &) = delete;
        /** Default move constructor */
        CpuDepthwiseConvolutionOptimizedInternal(CpuDepthwiseConvolutionOptimizedInternal &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConvolutionOptimizedInternal &operator=(const CpuDepthwiseConvolutionOptimizedInternal &) = delete;
        /** Default move assignment operator */
        CpuDepthwiseConvolutionOptimizedInternal &operator=(CpuDepthwiseConvolutionOptimizedInternal &&) = default;
        /** Default destructor */
        ~CpuDepthwiseConvolutionOptimizedInternal() = default;
        /** Initialize the function's source, destination, kernels and border_size.
         *
         * @param[in, out] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
         * @param[in]      weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
         * @param[in]      biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                         Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[out]     output  Destination tensor info. Data type supported: same as @p input.
         * @param[in]      info    Depthwise convolution meta-data.
         */
        void configure(ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *output, const ConvolutionInfo &info);

        /** Static function to check if given info will lead to a valid configuration of @ref CpuDepthwiseConvolution3x3
         *
         * @param[in] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
         * @param[in] weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
         * @param[in] biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                    Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[in] output  Destination tensor info. Data type supported: same as @p input.
         * @param[in] info    Depthwise convolution meta-data.
         *
         * @return a status
         */
        static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const ConvolutionInfo &info);

        // Inherited methods overriden:
        void run(ITensorPack &tensors) override;
        void prepare(ITensorPack &tensors) override;

    private:
        std::unique_ptr<CpuDepthwiseConvolutionAssemblyDispatch> _dwc_optimized_func{ nullptr };
        std::unique_ptr<CpuPermute>                              _permute_input{ nullptr };
        std::unique_ptr<CpuPermute>                              _permute_weights{ nullptr };
        std::unique_ptr<CpuPermute>                              _permute_output{ nullptr };
        std::unique_ptr<CpuActivation>                           _activationlayer_function{ nullptr };
        bool                                                     _has_bias{ false };
        bool                                                     _is_quantized{ false };
        bool                                                     _is_nchw{ true };
        bool                                                     _permute{ false };
        bool                                                     _is_activationlayer_enabled{ false };
        bool                                                     _is_prepared{ false };
    };

    /** Basic function to execute a generic depthwise convolution. This function calls the following kernel:
     *
     * -# @ref CpuDepthwiseConvolutionNativeKernel
     *
     */
    class CpuDepthwiseConvolutionGeneric : public ICpuOperator
    {
    public:
        /** Default constructor */
        CpuDepthwiseConvolutionGeneric();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConvolutionGeneric(const CpuDepthwiseConvolutionGeneric &) = delete;
        /** Default move constructor */
        CpuDepthwiseConvolutionGeneric(CpuDepthwiseConvolutionGeneric &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CpuDepthwiseConvolutionGeneric &operator=(const CpuDepthwiseConvolutionGeneric &) = delete;
        /** Default move assignment operator */
        CpuDepthwiseConvolutionGeneric &operator=(CpuDepthwiseConvolutionGeneric &&) = default;
        /** Default destructor */
        ~CpuDepthwiseConvolutionGeneric() = default;
        /** Initialize the function's source, destination, weights and convolution information.
         *
         * @param[in, out] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
         * @param[out]     output  Destination tensor info. Data type supported: same as @p input.
         * @param[in]      weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                         Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
         * @param[in]      biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                         Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[in]      info    Depthwise convolution meta-data.
         */
        void configure(ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *output, const ConvolutionInfo &info);

        /** Static function to check if given info will lead to a valid configuration of @ref CpuDepthwiseConvolutionGeneric
         *
         * @param[in] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
         * @param[in] output  Destination tensor info. Data type supported: same as @p input.
         * @param[in] weights Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                    Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
         * @param[in] biases  Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                    Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[in] info    Depthwise convolution meta-data.
         *
         * @return a status
         */
        static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const ConvolutionInfo &info);

        // Inherited methods overridden:
        void run(ITensorPack &tensors) override;
        void prepare(ITensorPack &tensors) override;

    private:
        std::unique_ptr<kernels::CpuDepthwiseConvolutionNativeKernel> _depthwise_conv_kernel{ nullptr };
        std::unique_ptr<CpuPermute>                                   _permute_input{ nullptr };
        std::unique_ptr<CpuPermute>                                   _permute_weights{ nullptr };
        std::unique_ptr<CpuPermute>                                   _permute_output{ nullptr };
        std::unique_ptr<CpuActivation>                                _activationlayer_function{ nullptr };
        bool                                                          _is_nchw{ true };
        bool                                                          _is_prepared{ false };
        bool                                                          _is_activationlayer_enabled{ false };
    };

    DepthwiseConvolutionFunction             _depth_conv_func;
    CpuDepthwiseConvolutionOptimizedInternal _func_optimized;
    CpuDepthwiseConvolutionGeneric           _func_generic;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DEQUANTIZATION_H */
