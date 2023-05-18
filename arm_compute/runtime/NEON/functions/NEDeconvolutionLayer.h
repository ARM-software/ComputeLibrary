/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_NEDECONVOLUTIONLAYER_H
#define ARM_COMPUTE_NEDECONVOLUTIONLAYER_H

#include "arm_compute/runtime/CPP/functions/CPPUpsample.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReverse.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
/** Function to run the deconvolution layer.
 *
 * Deconvolution Layer is the backward pass of Convolution Layer. First we transform the input depending on the stride and pad info and then perfrom a 1x1
 * convolution pass. Input stride defines how many zeroes we should put between each element of the input, pad is the amount of padding and finaly a is a user
 * specified value where a < stride - 1 that increases the padding top and right of the input image.
 *
 *  The relation between input to output is as follows:
 *  \f[
 *       width\_output = (width\_input - 1) \cdot stride\_x - 2 \cdot padding\_x + kernel\_x
 *  \f]
 *  \f[
 *       height\_output = (height\_input - 1) \cdot stride\_y - 2 \cdot padding\_y + kernel\_y
 *  \f]
 *
 *  where
 *      width is the size of the first input dimension.
 *      height is the size of the second input dimension.
 *      width_output is the size of the first output dimension.
 *      height_output is the size of the second output dimension.
 *      kernel_x and kernel_y are the convolution sizes in x and y.
 *      stride_x and stride_y is the input stride of the first and second dimension.
 *
 * The weights used by Deconvolution are supposed to be the same as the ones used for Convolution. Therefore, it will be necessary to use the weights in the
 * reverse order to perform an actual convolution. This is achieved by using @ref NEReverse.
 *
 * This function calls the following kernels/functions:
 *
 * -# @ref CPPUpsample
 * -# @ref NEConvolutionLayer
 * -# @ref NEReverse
 *
 */
class NEDeconvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDeconvolutionLayer(const NEDeconvolutionLayer &) = delete;
    /** Default move constructor */
    NEDeconvolutionLayer(NEDeconvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDeconvolutionLayer &operator=(const NEDeconvolutionLayer &) = delete;
    /** Default move assignment operator */
    NEDeconvolutionLayer &operator=(NEDeconvolutionLayer &&) = default;
    /** Default destructor */
    ~NEDeconvolutionLayer() = default;

    /** Set the input, weights, biases and output tensors.
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
     * @param[in,out] input            Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
     *                                 Data types supported: F32/F16/QASYMM8/QASYMM8_SIGNED.
     * @param[in]     weights          The 4d weights with dimensions [width, height, IFM, OFM].
     *                                 Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]     bias             Optional, ignored if NULL. The biases have one dimension.
     *                                 Data type supported: Data types supported: S32 for QASYMM8/QASYMM8_SIGNED input, F32 for F32 input, F16 for F16 input.
     * @param[out]    output           Output tensor. The output has the same number of dimensions as the @p input.
     * @param[in]     info             Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
     * @param[in]     enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                                 available which may introduce a drop of accuracy as well. Default is false
     * @param[in]     weights_info     (Optional) Specifies the weight format. Default is unspecified. This parameter can be used to specify the weight format that is optimal for
     *                                 the GEMM convolution.
     *
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info, bool enable_fast_math = false, const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEDeconvolutionLayer
     *
     * @param[in] input            Input tensor info. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
     *                             Data types supported: F32/F16/QASYMM8/QASYMM8_SIGNED.
     * @param[in] weights          The 4d weights info with dimensions [width, height, IFM, OFM].
     *                             Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] bias             (Optional) The biases have one dimension. Data type supported: Data types supported: S32 for QASYMM8/QASYMM8_SIGNED input, F32 for F32 input, F16 for F16 input.
     * @param[in] output           Output tensor info. The output has the same number of dimensions as the @p input.
     * @param[in] info             Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     * @param[in] weights_info     (Optional) Specifies the weight format. Default is unspecified. This parameter can be used to specify the weight format that is optimal for
     *                             the GEMM convolution.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &info,
                           bool enable_fast_math = false, const WeightsInfo &weights_info = WeightsInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup        _memory_group;
    NEConvolutionLayer _conv_f;
    CPPUpsample        _upsample_f;
    NEReverse          _flip_weights;
    Tensor             _scaled_output;
    Tensor             _weights_flipped;
    Tensor             _flip_axis;
    const ITensor     *_original_weights;
    ITensor           *_input;
    PadStrideInfo      _info;
    bool               _is_prepared;
    bool               _do_upsampling;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEDECONVOLUTIONLAYER_H */
