/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H
#define ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/ITransformWeights.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class CLCol2ImKernel;
class CLIm2ColKernel;
class CLWeightsReshapeKernel;
class ICLTensor;

/** Function to reshape and transpose the weights. This function calls the following kernels:
 * -# @ref CLWeightsReshapeKernel
 */
class CLConvolutionLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    CLConvolutionLayerReshapeWeights();
    /** Prevent instances of this class from being copied */
    CLConvolutionLayerReshapeWeights(const CLConvolutionLayerReshapeWeights &) = delete;
    /** Prevent instances of this class from being copied */
    CLConvolutionLayerReshapeWeights &operator=(const CLConvolutionLayerReshapeWeights &) = delete;
    /** Default move constructor */
    CLConvolutionLayerReshapeWeights(CLConvolutionLayerReshapeWeights &&) = default;
    /** Default move assignment operator */
    CLConvolutionLayerReshapeWeights &operator=(CLConvolutionLayerReshapeWeights &&) = default;
    /** Default destructor */
    ~CLConvolutionLayerReshapeWeights();
    /** Set the input and output tensors.
     *
     * @param[in]  weights    Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                        Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[in]  biases     Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output     Destination tensor. Data types supported: Same as @p weights.
     * @param[in]  num_groups (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, unsigned int num_groups = 1);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  weights         Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[in]  biases          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output          Destination tensor. Data types supported: Same as @p weights.
     * @param[in]  num_groups      (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvolutionLayerReshapeWeights
     *
     * @param[in] weights    Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                       Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[in] biases     Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[in] output     Destination tensor. Data types supported: Same as @p weights.
     * @param[in] num_groups (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, unsigned int num_groups = 1);
    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<CLWeightsReshapeKernel> _weights_reshape_kernel;
};

namespace weights_transformations
{
/** Basic function to manage the reshape weights generated from @ref CLConvolutionLayerReshapeWeights */
class CLConvolutionLayerReshapeWeightsTransform : public ITransformWeights
{
public:
    /** Configures the @ref CLConvolutionLayerReshapeWeights function
     *
     * @param[in] input      Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[in] biases     Biases tensor. Data type supported: same as @p input, S32 if @p input is quantized.
     * @param[in] num_groups Number of groups when performing a grouped convolution.
     */
    void configure(const ICLTensor *input, const ICLTensor *biases, unsigned int num_groups)
    {
        configure(CLKernelLibrary::get().get_compile_context(), input, biases, num_groups);
    }
    /** Configures the @ref CLConvolutionLayerReshapeWeights function
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] input           Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[in] biases          Biases tensor. Data type supported: same as @p input, S32 if @p input is quantized.
     * @param[in] num_groups      Number of groups when performing a grouped convolution.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *biases, unsigned int num_groups)
    {
        _bias_bit   = (biases != nullptr) ? 1 : 0;
        _num_groups = num_groups;
        _func.configure(compile_context, input, biases, &_output, num_groups);
    }

    //Inherited method override
    void run() override
    {
        _output.allocator()->allocate();
        _func.run();
        _reshape_run = true;
    }

    //Inherited method override
    ICLTensor *get_weights() override
    {
        return &_output;
    }

    //Inherited method override
    void release() override
    {
        _output.allocator()->free();
    }

    //Inherited method override
    uint32_t uid() override
    {
        return ((0x9) | (_bias_bit << 7) | (_num_groups << 8));
    }

private:
    CLTensor                         _output{};
    CLConvolutionLayerReshapeWeights _func{};
    int32_t                          _bias_bit{ 0 };
    unsigned int                     _num_groups{ 0 };
};
} // namespace weights_transformations

/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref CLIm2ColKernel
 * -# @ref CLGEMM (if the data type is FP32 or FP16)
 * -# @ref CLGEMMLowpMatrixMultiplyCore (if the data type is QASYMM8/QASYMM8_SIGNED)
 * -# @ref CLGEMMLowpOutputStage with QUANTIZE_DOWN_FIXEDPOINT type of quantization (if the data type is QASYMM8/QASYMM8_SIGNED)
 * -# @ref CLCol2ImKernel (if NCHW data layout)
 */
class CLGEMMConvolutionLayer : public IFunction
{
public:
    /** Constructor
     *
     * @param[in] memory_manager  (Optional) Memory manager.
     * @param[in] weights_manager (Optional) Weights manager.
     */
    CLGEMMConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMConvolutionLayer(const CLGEMMConvolutionLayer &) = delete;
    /** Default move constructor */
    CLGEMMConvolutionLayer(CLGEMMConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMConvolutionLayer &operator=(const CLGEMMConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLGEMMConvolutionLayer &operator=(CLGEMMConvolutionLayer &&) = default;
    /**Default destructor */
    ~CLGEMMConvolutionLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2     |dst            |
     * |:--------------|:------------------|:--------|:--------------|
     * |F16            |F16                |F16      |F16            |
     * |F32            |F32                |F32      |F32            |
     * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                          Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights         Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]  biases          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[out] output          Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info    Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                             tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation        (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups      (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                          Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param[in]      compile_context       The compile context to be used.
     * @param[in]      input                 Input tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]      weights               Weights tensor. Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or
     *                                       QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]      biases                Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                                       Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[in, out] output                Output tensor. Data types supported: same as @p input.
     * @param[in]      gemmlowp_output_stage GEMMLowp output stage info
     * @param[in]      gemm_3d_depth         Depth of GEMM 3D
     * @param[in]      act_info              Activation to apply after the matrix multiplication
     */
    void configure_mm(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                      const GEMMLowpOutputStageInfo &gemmlowp_output_stage,
                      int gemm_3d_depth, const ActivationLayerInfo &act_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer matrix multiply routines
     *
     * @param[in] input                 Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights               Weights tensor info. Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or
     *                                  QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in] biases                Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                                  Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[in] output                Output tensor info. Data types supported: same as @p input.
     * @param[in] gemmlowp_output_stage GEMMLowp output stage info
     * @param[in] gemm_3d_depth         Depth of GEMM 3D
     * @param[in] skip_im2col           Flag which specifies if im2col has to be skipped. i.e. 1x1 convolution with NHWC data layout.
     * @param[in] act_info              Activation to apply after the matrix multiplication
     *
     * @return a status
     */
    static Status validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const GEMMLowpOutputStageInfo &gemmlowp_output_stage,
                              int gemm_3d_depth, bool skip_im2col, const ActivationLayerInfo &act_info);

private:
    MemoryGroup                                                        _memory_group;
    IWeightsManager                                                   *_weights_manager;
    CLConvolutionLayerReshapeWeights                                   _reshape_weights;
    weights_transformations::CLConvolutionLayerReshapeWeightsTransform _reshape_weights_managed;
    std::unique_ptr<CLIm2ColKernel>                                    _im2col_kernel;
    CLGEMM                                                             _mm_gemm;
    CLGEMMLowpMatrixMultiplyCore                                       _mm_gemmlowp;
    std::unique_ptr<CLCol2ImKernel>                                    _col2im_kernel;
    CLActivationLayer                                                  _activationlayer_function;

    const ICLTensor *_original_weights;

    CLTensor _im2col_output;
    CLTensor _weights_reshaped;
    CLTensor _gemm_output;

    bool _skip_im2col;
    bool _skip_col2im;
    bool _is_quantized;
    bool _fuse_activation;
    bool _is_prepared;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H */
