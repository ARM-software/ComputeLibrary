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
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to simulate a convolution layer. This function calls one of the following functions:
 * -# @ref CpuGemm     (executed only in case GEMM is required for the operation)
 * -# @ref CpuWinogradConv2d (executed only in case Winograd is required for the operation)
 * -# @ref CpuDirectConv2d   (executed only in case Direct Convolution is required for the operation)
 *
 *
 * The function selects one of the algorithms mentioned above based on:
 *      - The size of the kernel
 *      - Number of input/output feature maps
 *      - Amount of memory needed
 *
 * Generally GEMM-based convolution is executed when neither Winograd nor FFT nor Direct convolution can be performed.
 *
 * FP32 Algorithm| Filter Size                                        |   Input/Output feature maps               |
 * --------------|----------------------------------------------------|-------------------------------------------|
 * Winograd      | 3x3 1x3 3x1 5x1 1x5 5x5(fast maths) 7x1 1x7        |  Input channels is greater than 3         |
 * FFT           | Squared kernels and greater than 9x9               |  Input feature maps > Output feature maps |
 * DirectConv    | 9x9                                                |                                           |
 * GEMM          | Any size                                           |                                           |
 *
 * Winograd 5x5 requires fast maths enabled.
 *
 * FP16 Algorithm| Filter Size      |
 * --------------|------------------|
 * Winograd      | Not supported    |
 * FFT           | Not supported    |
 * DirectConv    | 9x9              |
 * GEMM          | Any size         |
 *
 *
 */
class CpuConv2d : public ICpuOperator
{
public:
    /** Constructor */
    CpuConv2d();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuConv2d);
    /** Default destructor */
    ~CpuConv2d();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
     *
     * @param[in]  src              Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights          Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                              Data type supported: Same as @p src, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  biases           Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Same as @p src, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] dst              Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p src.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                              tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(ITensorInfo *src, ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuConv2d
     *
     * Similar to CpuConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false,
                           unsigned int num_groups = 1);
    /** Static function to check if given info will return the convolution called by @ref CpuConv2d
     *
     * @param[in] src              Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights          Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported:Same as @p src, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] dst              Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p src.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                             tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     *
     * @return the Convolution Method Hint
     */
    static ConvolutionMethod get_convolution_method(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                                                    const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    std::unique_ptr<ICpuOperator>    _function;
    experimental::MemoryRequirements _aux_mem{};
};
} // namespace cpu
} // namespace arm_compute
