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
#ifndef ARM_COMPUTE_CL_GEMM_CONV2D_H
#define ARM_COMPUTE_CL_GEMM_CONV2D_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

#include <memory>

namespace arm_compute
{
namespace opencl
{
class ClGemm;
class ClGemmLowpMatrixMultiplyCore;
namespace kernels
{
class ClIm2ColKernel;
class ClCol2ImKernel;
class ClWeightsReshapeKernel;
class ClActivationKernel;
} // namespace kernels

/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref opencl::kernels::ClIm2ColKernel
 * -# @ref ClGemm (if the data type is FP32 or FP16)
 * -# @ref CLGEMMLowpMatrixMultiplyCore (if the data type is QASYMM8/QASYMM8_SIGNED)
 * -# @ref ClGemmLowpOutputStage with QUANTIZE_DOWN_FIXEDPOINT type of quantization (if the data type is QASYMM8/QASYMM8_SIGNED)
 * -# @ref opencl::kernels::ClCol2ImKernel (if NCHW data layout)
 * -# @ref opencl::kernels::ClActivationKernel
 */
class ClGemmConv2d : public IClOperator
{
public:
    /** Constructor */
    ClGemmConv2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ClGemmConv2d(const ClGemmConv2d &) = delete;
    /** Default move constructor */
    ClGemmConv2d(ClGemmConv2d &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ClGemmConv2d &operator=(const ClGemmConv2d &) = delete;
    /** Default move assignment operator */
    ClGemmConv2d &operator=(ClGemmConv2d &&) = default;
    /**Default destructor */
    ~ClGemmConv2d();
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
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights         Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]  biases          Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[out] dst             Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in]  conv2d_info     Contains convolution 2d info described in @ref Conv2dInfo.
     * @param[in]  weights_info    Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                             tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                   const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClGemmConvolution::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const Conv2dInfo &conv2d_info,
                           const WeightsInfo &weights_info = WeightsInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param[in]      compile_context       The compile context to be used.
     * @param[in]      src                   Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]      weights               Weights tensor info. Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or
     *                                       QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in]      biases                Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                                       Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[in, out] dst                   Output tensor info. Data types supported: same as @p input.
     * @param[in]      gemmlowp_output_stage GEMMLowp output stage info
     * @param[in]      gemm_3d_depth         Depth of GEMM 3D
     * @param[in]      act_info              Activation to apply after the matrix multiplication
     */
    void configure_mm(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                      const GEMMLowpOutputStageInfo &gemmlowp_output_stage,
                      int gemm_3d_depth, const ActivationLayerInfo &act_info, const experimental::PostOpList<ITensorInfo *> &post_ops = experimental::PostOpList<ITensorInfo *> {});
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer matrix multiply routines
     *
     * @param[in] src                   Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights               Weights tensor info. Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8 or
     *                                  QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8_SIGNED.
     * @param[in] biases                Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                                  Data type supported: Should match @p input data type, except for input of quantized type where biases should be of S32 type.
     * @param[in] dst                   Output tensor info. Data types supported: same as @p input.
     * @param[in] gemmlowp_output_stage GEMMLowp output stage info
     * @param[in] gemm_3d_depth         Depth of GEMM 3D
     * @param[in] skip_im2col           Flag which specifies if im2col has to be skipped. i.e. 1x1 convolution with NHWC data layout.
     * @param[in] act_info              Activation to apply after the matrix multiplication
     *
     * @return a status
     */
    static Status validate_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const GEMMLowpOutputStageInfo &gemmlowp_output_stage,
                              int gemm_3d_depth, bool skip_im2col, const ActivationLayerInfo &act_info, const experimental::PostOpList<ITensorInfo *> &post_ops = experimental::PostOpList<ITensorInfo *> {});

    enum AuxTensorIdx
    {
        // ClGemmLowpMatrixMultiplyCore has up to 7 internal tensors
        Im2ColOutput = 8,
        WeightsReshaped,
        GemmOutput,
        Count
    };

    std::unique_ptr<kernels::ClWeightsReshapeKernel> _weights_reshape_kernel;
    std::unique_ptr<kernels::ClIm2ColKernel>         _im2col_kernel;
    std::unique_ptr<ClGemm>                          _mm_gemm;
    std::unique_ptr<ClGemmLowpMatrixMultiplyCore>    _mm_gemmlowp;
    std::unique_ptr<opencl::kernels::ClCol2ImKernel> _col2im_kernel;
    std::unique_ptr<kernels::ClActivationKernel>     _activation_kernel;

    TensorInfo _im2col_output;
    TensorInfo _weights_reshaped;
    TensorInfo _gemm_output;

    bool _skip_im2col;
    bool _skip_col2im;
    bool _is_quantized;
    bool _fuse_activation;
    bool _append_bias;
    bool _is_prepared;
    bool _use_post_ops;

    experimental::MemoryRequirements _aux_mem;
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMM_CONV2D_H */
