/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H
#define ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H

#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NCHWKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NHWCKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayerNativeKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayerReshapeWeightsKernel.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/ICLDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLPermute.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class ICLTensor;

/** Function to execute a depthwise convolution
 */
class CLDepthwiseConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer(const CLDepthwiseConvolutionLayer &) = delete;
    /** Default move constructor */
    CLDepthwiseConvolutionLayer(CLDepthwiseConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer &operator=(const CLDepthwiseConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLDepthwiseConvolutionLayer &operator=(CLDepthwiseConvolutionLayer &&) = default;
    /** Initialize the function's source, destination, weights and convolution information.
     *
     * @param[in, out] input            Source tensor. Data type supported: QASYMM8/FP16/FP32. Data layout supported: NHWC, NCHW
     * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                                  Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                                  Data type supported: Same as @p input, S32 when input is QASYMM8.
     * @param[out]     output           Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info        Padding and stride information to use for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                   ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));
    /** Initialize the function's source, destination, weights and convolution information.
     *
     * @param[in]      compile_context  The compile context to be used.
     * @param[in, out] input            Source tensor. Data type supported: QASYMM8/FP16/FP32. Data layout supported: NHWC, NCHW
     * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                                  Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                                  Data type supported: Same as @p input, S32 when input is QASYMM8.
     * @param[out]     output           Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info        Padding and stride information to use for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayer
     *
     * @param[in] input            Source tensor info. Data type supported: QASYMM8/FP16/FP32. Data layout supported: NHWC, NCHW
     * @param[in] weights          Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in] biases           Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                             Data type supported: Same as @p input, S32 when input is QASYMM8.
     * @param[in] output           Destination tensor. Data type supported: same as @p input.
     * @param[in] conv_info        Padding and stride information to use for the convolution.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                           ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    // Inherited methods overriden:
    void run() override;
    void prepare() override;

private:
    /** Static function to choose the best depthwise convolution function for @ref CLDepthwiseConvolutionLayer
     *
     * @param[in] input            Source tensor info. Data type supported: QASYMM8/FP16/FP32. Data layout supported: NHWC, NCHW
     * @param[in] weights          Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in] biases           Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                             Data type supported: Same as @p input, S32 when input is QASYMM8.
     * @param[in] output           Destination tensor. Data type supported: same as @p input.
     * @param[in] conv_info        Padding and stride information to use for the convolution.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] gpu_target       (Optional) GPU target to validate the kernel for. Defaults to midgard.
     *
     * @return a Depthwise Convolution Function
     */
    static DepthwiseConvolutionFunction get_depthwiseconvolution_function(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                                          const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                                                                          ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U), GPUTarget gpu_target = GPUTarget::MIDGARD);

    /** Basic function to execute a depthwise convolution for kernel size 3x3xC (when data layout NCHW) or Cx3x3 (when data layout NHWC). This function calls the following OpenCL kernels:
    *
    * -# @ref CLDepthwiseConvolutionLayer3x3NCHWKernel (if data_layout == NCHW)
    * -# @ref CLDepthwiseConvolutionLayer3x3NHWCKernel (if data_layout == NHWC)
    * -# @ref CLDepthwiseConvolutionLayerReshapeWeightsKernel (if data_layout == NHWC)
    * -# @ref CLFillBorderKernel (if pad_x or pad_y > 0)
    *
    */
    class CLDepthwiseConvolutionLayerInternal3x3 : public IFunction
    {
    public:
        /** Default constructor */
        CLDepthwiseConvolutionLayerInternal3x3(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CLDepthwiseConvolutionLayerInternal3x3(const CLDepthwiseConvolutionLayerInternal3x3 &) = delete;
        /** Default move constructor */
        CLDepthwiseConvolutionLayerInternal3x3(CLDepthwiseConvolutionLayerInternal3x3 &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CLDepthwiseConvolutionLayerInternal3x3 &operator=(const CLDepthwiseConvolutionLayerInternal3x3 &) = delete;
        /** Default move assignment operator */
        CLDepthwiseConvolutionLayerInternal3x3 &operator=(CLDepthwiseConvolutionLayerInternal3x3 &&) = default;
        /** Initialize the function's source, destination, conv and border_size.
         *
         * @param[in, out] input            Source tensor. Data type supported: QASYMM8/F16/F32. (Written to only for border filling).
         * @param[in]      weights          Weights tensor. A 3D tensor with shape [3, 3, IFM].
         *                                  Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
         * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                                  Data type supported: Same as @p input.
         * @param[out]     output           Destination tensor. Data type supported: same as @p input.
         * @param[in]      conv_info        Padding and stride information to use for the convolution.
         * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
         * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         */
        void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                       ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));
        /** Initialize the function's source, destination, conv and border_size.
         *
         * @param[in]      compile_context  The compile context to be used.
         * @param[in, out] input            Source tensor. Data type supported: QASYMM8/F16/F32. (Written to only for border filling).
         * @param[in]      weights          Weights tensor. A 3D tensor with shape [3, 3, IFM].
         *                                  Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
         * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                                  Data type supported: Same as @p input.
         * @param[out]     output           Destination tensor. Data type supported: same as @p input.
         * @param[in]      conv_info        Padding and stride information to use for the convolution.
         * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
         * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         */
        void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                       unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

        /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayer3x3
         *
         * @param[in] input            Source tensor info. Data type supported: QASYMM8 for all layouts, F16/F32 for NCHW.
         * @param[in] weights          Weights tensor info. A 3D tensor with shape [3, 3, IFM].
         *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
         * @param[in] biases           Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                             Data type supported: Same as @p input, S32 when input is QASYMM8.
         * @param[in] output           Destination tensor. Data type supported: same as @p input.
         * @param[in] conv_info        Padding and stride information to use for the convolution.
         * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
         * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         *
         * @return a status
         */
        static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                               ActivationLayerInfo act_info = ActivationLayerInfo(), GPUTarget gpu_target = GPUTarget::MIDGARD, const Size2D &dilation = Size2D(1U, 1U));

        // Inherited methods overriden:
        void run() override;
        void prepare() override;

        void set_memory_group(std::shared_ptr<IMemoryManager> memory_manager)
        {
            _memory_group = MemoryGroup(std::move(memory_manager));
        };

    private:
        MemoryGroup                                            _memory_group;
        std::unique_ptr<ICLDepthwiseConvolutionLayer3x3Kernel> _kernel;
        CLFillBorderKernel                                     _border_handler;
        CLPermute                                              _permute_input_to_nchw;
        CLPermute                                              _permute_weights_to_nchw;
        CLPermute                                              _permute_output_to_nhwc;
        CLDepthwiseConvolutionLayerReshapeWeightsKernel        _reshape_weights;
        CLTensor                                               _permuted_input;
        CLTensor                                               _permuted_weights;
        CLTensor                                               _permuted_output;
        CLTensor                                               _output_multipliers;
        CLTensor                                               _output_shifts;
        const ITensor                                         *_original_weights;
        const ITensor                                         *_input;
        const ITensor                                         *_output;
        bool                                                   _needs_permute;
        bool                                                   _needs_weights_reshape;
        bool                                                   _is_prepared;
        bool                                                   _is_quantized;
    };

    /** Basic function to execute a generic depthwise convolution. This function calls the following OpenCL kernels:
     *
     * -# @ref CLDepthwiseConvolutionLayerNativeKernel
     * -# @ref CLPermute (x 3) if the data layout is NCHW
     *
     */
    class CLDepthwiseConvolutionLayerGeneric : public IFunction
    {
    public:
        /** Default constructor */
        CLDepthwiseConvolutionLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CLDepthwiseConvolutionLayerGeneric(const CLDepthwiseConvolutionLayerGeneric &) = delete;
        /** Default move constructor */
        CLDepthwiseConvolutionLayerGeneric(CLDepthwiseConvolutionLayerGeneric &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        CLDepthwiseConvolutionLayerGeneric &operator=(const CLDepthwiseConvolutionLayerGeneric &) = delete;
        /** Default move assignment operator */
        CLDepthwiseConvolutionLayerGeneric &operator=(CLDepthwiseConvolutionLayerGeneric &&) = default;
        /** Initialize the function's source, destination, weights and convolution information.
         *
         * @param[in, out] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F32. (Written to only for border filling).
         * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                                  Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
         * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                                  Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[out]     output           Destination tensor. Data type supported: same as @p input.
         * @param[in]      conv_info        Padding and stride information to use for the convolution.
         * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
         * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         */
        void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                       unsigned int depth_multiplier = 1, const ActivationLayerInfo &act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));
        /** Initialize the function's source, destination, weights and convolution information.
         *
         * @param[in]      compile_context  The compile context to be used.
         * @param[in, out] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F32. (Written to only for border filling).
         * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                                  Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
         * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                                  Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[out]     output           Destination tensor. Data type supported: same as @p input.
         * @param[in]      conv_info        Padding and stride information to use for the convolution.
         * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
         * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         */
        void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                       unsigned int depth_multiplier = 1, const ActivationLayerInfo &act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

        /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayerGeneric
         *
         * @param[in] input            Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F32.
         * @param[in] weights          Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                             Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
         * @param[in] biases           Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                             Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[in] output           Destination tensor. Data type supported: same as @p input.
         * @param[in] conv_info        Padding and stride information to use for the convolution.
         * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
         * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         *
         * @return a status
         */
        static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                               unsigned int depth_multiplier = 1, const ActivationLayerInfo &act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

        // Inherited methods overriden:
        void run() override;
        void prepare() override;

        void set_memory_group(std::shared_ptr<IMemoryManager> memory_manager)
        {
            _memory_group = MemoryGroup(std::move(memory_manager));
        };

    private:
        MemoryGroup _memory_group;

        CLDepthwiseConvolutionLayerNativeKernel _dwc_native_kernel;
        CLPermute                               _permute_input_to_nhwc;
        CLPermute                               _permute_weights_to_nhwc;
        CLPermute                               _permute_output_to_nchw;

        CLTensor       _permuted_input;
        CLTensor       _permuted_weights;
        CLTensor       _permuted_output;
        CLTensor       _output_multipliers;
        CLTensor       _output_shifts;
        const ITensor *_original_weights;
        const ITensor *_input;
        const ITensor *_output;

        bool _needs_permute;
        bool _is_prepared;
        bool _is_quantized;
    };

    std::shared_ptr<IMemoryManager> _memory_manager;

    DepthwiseConvolutionFunction           _depth_conv_func;
    CLDepthwiseConvolutionLayerInternal3x3 _func_3x3;
    CLDepthwiseConvolutionLayerGeneric     _func_generic;
};

/** Basic function to execute a depthwise convolution for kernel size 3x3xC (when data layout NCHW) or Cx3x3 (when data layout NHWC). This function calls the following OpenCL kernels:
 *
 * -# @ref CLDepthwiseConvolutionLayer3x3NCHWKernel (if data_layout == NCHW)
 * -# @ref CLDepthwiseConvolutionLayer3x3NHWCKernel (if data_layout == NHWC)
 * -# @ref CLDepthwiseConvolutionLayerReshapeWeightsKernel (if data_layout == NHWC)
 * -# @ref CLFillBorderKernel (if pad_x or pad_y > 0)
 *
 */
class CLDepthwiseConvolutionLayer3x3 : public IFunction
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer3x3(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer3x3(const CLDepthwiseConvolutionLayer3x3 &) = delete;
    /** Default move constructor */
    CLDepthwiseConvolutionLayer3x3(CLDepthwiseConvolutionLayer3x3 &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer3x3 &operator=(const CLDepthwiseConvolutionLayer3x3 &) = delete;
    /** Default move assignment operator */
    CLDepthwiseConvolutionLayer3x3 &operator=(CLDepthwiseConvolutionLayer3x3 &&) = default;
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in, out] input            Source tensor. Data type supported: QASYMM8/F16/F32. (Written to only for border filling).
     * @param[in]      weights          Weights tensor. A 3D tensor with shape [3, 3, IFM].
     *                                  Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                                  Data type supported: Same as @p input.
     * @param[out]     output           Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info        Padding and stride information to use for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
     * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    ARM_COMPUTE_DEPRECATED_REL_REPLACE(20.02, CLDepthwiseConvolutionLayer)
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                   ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]      compile_context  The compile context to be used.
     * @param[in, out] input            Source tensor. Data type supported: QASYMM8/F16/F32. (Written to only for border filling).
     * @param[in]      weights          Weights tensor. A 3D tensor with shape [3, 3, IFM].
     *                                  Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                                  Data type supported: Same as @p input.
     * @param[out]     output           Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info        Padding and stride information to use for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
     * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    ARM_COMPUTE_DEPRECATED_REL_REPLACE(20.02, CLDepthwiseConvolutionLayer)
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayer3x3
     *
     * @param[in] input            Source tensor info. Data type supported: QASYMM8 for all layouts, F16/F32 for NCHW.
     * @param[in] weights          Weights tensor info. A 3D tensor with shape [3, 3, IFM].
     *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in] biases           Biases tensor info. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                             Data type supported: Same as @p input, S32 when input is QASYMM8.
     * @param[in] output           Destination tensor. Data type supported: same as @p input.
     * @param[in] conv_info        Padding and stride information to use for the convolution.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for 3x3 QASYMM8 supported.
     * @param[in] gpu_target       (Optional) GPU target to validate the kernel for. Defaults to midgard.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                           ActivationLayerInfo act_info = ActivationLayerInfo(), GPUTarget gpu_target = GPUTarget::MIDGARD, const Size2D &dilation = Size2D(1U, 1U));

    // Inherited methods overriden:
    void run() override;
    void prepare() override;

private:
    CLDepthwiseConvolutionLayer _func;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H */
