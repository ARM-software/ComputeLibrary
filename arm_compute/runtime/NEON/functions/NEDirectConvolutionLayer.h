/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYER_H__

#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerOutputStageKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
/** Function to run the direct convolution.
 *
 *  This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel for the input
 * -# @ref NEDirectConvolutionLayerOutputStageKernel
 * -# @ref NEDirectConvolutionLayerKernel
 */
class NEDirectConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEDirectConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input, weights, biases and output tensors.
     *
     * @note: DirectConvolution only works in the following configurations:
     *    1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = QS8/QS16/F16/F32
     *    3x3 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = QS8/F16/F32
     *    5x5 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F32
     *
     * @param[in, out] input     Input tensor. Data types supported: QS8/QS16/F16/F32.
     * @param[in]      weights   Set of kernels to convolve the input volume.
     *                           Supported sizes: 1x1, 3x3 and 5x5.
     *                           The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                           Data type supported: Same as @p input.
     * @param[in]      bias      Set of biases. Can be nullptr. Data type supported: Same as @p input.
     * @param[out]     output    Output tensor.
     *                           The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
     * @param[in]      conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDirectConvolutionLayer
     *
     * @note: DirectConvolution only works in the following configurations:
     *    1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = QS8/QS16/F16/F32
     *    3x3 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = QS8/F16/F32
     *    5x5 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F32
     *
     * @param[in] input     Input tensor. Data types supported: QS8/QS16/F16/F32.
     * @param[in] weights   Set of kernels to convolve the input volume.
     *                      Supported sizes: 1x1, 3x3 and 5x5.
     *                      The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                      Data type supported: Same as @p input.
     * @param[in] bias      Set of biases. Can be nullptr. Data type supported: Same as @p input.
     * @param[in] output    Output tensor.
     *                      The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                               _memory_group;
    NEDirectConvolutionLayerOutputStageKernel _output_stage_kernel;
    NEDirectConvolutionLayerKernel            _conv_kernel;
    NEFillBorderKernel                        _input_border_handler;
    Tensor                                    _accumulator;
    bool                                      _has_bias;
    bool                                      _is_fixed_point;
};
}
#endif /* __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYER_H__ */
