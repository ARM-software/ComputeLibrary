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
#ifndef ARM_COMPUTE_CPU_GEMM_CONV2D_H
#define ARM_COMPUTE_CPU_GEMM_CONV2D_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "src/cpu/ICpuOperator.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
class CpuGemm;
class CpuGemmLowpMatrixMultiplyCore;
class CpuGemmLowpOutputStage;
namespace kernels
{
class CpuWeightsReshapeKernel;
class CpuIm2ColKernel;
class CpuCol2ImKernel;
class CpuReshapeKernel;
} // namespace kernels

/** Basic function to compute the convolution layer. This function calls the following kernels/functions:
 *
 * -# @ref cpu::kernels::CpuIm2ColKernel
 * -# @ref CpuGemm (if the data type is BFLOAT16/FP16/FP32)
 * -# @ref CpuGemmLowpMatrixMultiplyCore (if the data type is QASYMM8/QASYMM8_SIGNED)
 * -# @ref CpuGemmLowpOutputStage (if the data type is QASYMM8/QASYMM8_SIGNED)
 * -# @ref cpu::kernels::CpuCol2ImKernel (if NCHW data layout)
 * -# @ref kernels::CpuWeightsReshapeKernel
 *
 */
class CpuGemmConv2d : public ICpuOperator
{
public:
    /** Constructor */
    CpuGemmConv2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGemmConv2d(const CpuGemmConv2d &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CpuGemmConv2d(CpuGemmConv2d &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGemmConv2d &operator=(const CpuGemmConv2d &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CpuGemmConv2d &operator=(CpuGemmConv2d &&) = delete;
    /** Destructor */
    ~CpuGemmConv2d();
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
     * |BFLOAT16       |BFLOAT16           |BFLOAT16 |BFLOAT16       |
     * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
     *
     * @param[in]  src              Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in]  weights          Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                              Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in]  biases           Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] dst              Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                              tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmConvolution::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                           bool enable_fast_math = false, unsigned int num_groups = 1);

    /** Indicates whether or not there is an optimal assembly implementation that can be used to process the given parameters.
     *
     * The paramter list is the same as @ref NEGEMMConvolutionLayer::has_opt_impl
     *
     * @return a status.
     */
    static Status has_opt_impl(arm_compute::WeightFormat &expected_weight_format, const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                               const PadStrideInfo &conv_info,
                               const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                               const bool enable_fast_math = false);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param[in]  src              Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in]  weights          Weights tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in]  biases           Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] dst              Output tensor info. Data types supported: Same as @p input,
     *                              except for input of QASYMM8/QASYMM8_SIGNED type where output should be of S32 type.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  gemm_3d_depth    (Optional) Depth of GEMM 3D (Defaults to 1)
     * @param[in]  fixed_format     (Optional) Select GEMM execution with variable weights.
     * @param[in]  weight_format    (Optional) The layout to be used for the weights tensor when running GEMM with variable weights.
     */
    void configure_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                      bool enable_fast_math = false, int gemm_3d_depth = 1, bool fixed_format = false, arm_compute::WeightFormat weight_format = arm_compute::WeightFormat::UNSPECIFIED);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConvolutionLayer matrix multiply routines
     *
     * @param[in] src              Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in] weights          Weights tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in] biases           Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] dst              Output tensor info. Data types supported: Same as @p input,
     *                             except for input of QASYMM8/QASYMM8_SIGNED type where output should be of S32 type.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     * @param[in] gemm_3d_depth    (Optional) Depth of GEMM 3D (Defaults to 1)
     * @param[in] skip_im2col      (Optional) Flag which specifies if im2col has to be skipped. i.e. 1x1 convolution with NHWC data layout. (Default to false)
     * @param[in] fixed_format     (Optional) Select GEMM execution with variable weights.
     * @param[in] weight_format    (Optional) The layout to be used for the weights tensor when running GEMM with variable weights.
     *
     * @return a status
     */
    static Status validate_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                              bool enable_fast_math = false, int gemm_3d_depth = 1, bool skip_im2col = false, bool fixed_format = false, arm_compute::WeightFormat weight_format = arm_compute::WeightFormat::UNSPECIFIED);
    /** Static function to check if GEMM3D is supported in @ref NEGEMM or in @ref CpuGemmMLowpMatrixMultiplyCore
     *
     * @param[in] src           Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in] weights       Weights tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in] act_info      Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in] gemm_3d_depth Depth of GEMM 3D
     * @param[in] skip_im2col   Flag which specifies if im2col has to be skipped. i.e. 1x1 convolution with NHWC data layout
     *
     * @return a status
     */
    static Status validate_gemm3d(const ITensorInfo *src, const ITensorInfo *weights, const ActivationLayerInfo &act_info, int gemm_3d_depth, bool skip_im2col);

    struct SkipInfo
    {
        bool skip_im2col;
        bool skip_col2im;
    };

    /** Static function to provide skip_im2col and skip_col2im information.
     *
     * @param[in] src       Input tensor info.
     * @param[in] weights   Weights tensor info.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] dilation  Dilation, in elements, across x and y.
     * @param[in] act_info  Activation layer information in case of a fused activation.
     *
     * @return a SkipInfo instance.
     */
    static SkipInfo skip_im_col_info(const ITensorInfo *src, const ITensorInfo *weights, const PadStrideInfo &conv_info,
                                     const Size2D &dilation, const ActivationLayerInfo &act_info);

    /** Indicates if the convolution executes in variable weights mode.
     *
     * Similar to @ref CpuGemm::isVarWeightsKernel
     */
    bool isVarWeightsKernel() const;
    enum AuxTensorIdx
    {
        // CpuGemmLowpMatrixMultiplyCore has up to 8 internal tensors
        Im2ColOutput = 9,
        WeightsReshaped,
        GemmOutput,
        Count
    };

    std::unique_ptr<kernels::CpuWeightsReshapeKernel> _weights_reshape_kernel;
    std::unique_ptr<cpu::kernels::CpuIm2ColKernel>    _im2col_kernel;
    std::unique_ptr<CpuGemm>                          _mm_gemm;
    std::unique_ptr<CpuGemmLowpMatrixMultiplyCore>    _mm_gemmlowp;
    std::unique_ptr<kernels::CpuCol2ImKernel>         _col2im_kernel;
    std::unique_ptr<kernels::CpuReshapeKernel>        _reshape_kernel;

    TensorInfo _im2col_output;
    TensorInfo _weights_reshaped;
    TensorInfo _gemm_output;
    TensorInfo _gemm_output_3d;

    DataLayout _data_layout;

    bool _skip_im2col;
    bool _skip_col2im;
    bool _is_quantized;
    bool _is_prepared;

    experimental::MemoryRequirements _aux_mem{ Count };
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMM_CONV2D_H */
