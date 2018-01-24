/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYEROUTPUTSTAGEKERNEL_H__
#define __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYEROUTPUTSTAGEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;
/** NEON kernel to accumulate the biases, if provided, or downscale in case of quantized input.
 *
 * @note We assume bias to be shared
 */
class NEDirectConvolutionLayerOutputStageKernel : public INEKernel
{
public:
    /** Default constructor */
    NEDirectConvolutionLayerOutputStageKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerOutputStageKernel(const NEDirectConvolutionLayerOutputStageKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerOutputStageKernel &operator=(const NEDirectConvolutionLayerOutputStageKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerOutputStageKernel(NEDirectConvolutionLayerOutputStageKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerOutputStageKernel &operator=(NEDirectConvolutionLayerOutputStageKernel &&) = default;
    /** Default destructor */
    ~NEDirectConvolutionLayerOutputStageKernel() = default;
    /** Set the accumulate buffer and the biases of the kernel.
     *
     * @param[in, out] input                        Input to add the bias to. If @p output is not specified then accumulation is done in-place.
     *                                              Data type supported: QS16/QS32/F16/F32
     * @param[in]      bias                         (Optional) The shared bias tensor to add. It must be 1D Tensor. Data type supported: Same as @p input
     * @param[out]     output                       (Optional) If the output tensor is specified the accumulation is done out-of-place. (Defaults to nullptr)
     *                                              Data type supported: QS8/QS16/F16/F32
     * @param[in]      result_fixedpoint_multiplier (Optional)Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]      result_shift                 (Optional)Integer value used to round to nearest division by a power-of-two the result after the fixed point multiplication
     * @param[in]      result_offset_after_shift    (Optional)Offset to be applied to result before converting it back to QASYMM8
     */
    void configure(ITensor *input, const ITensor *bias = nullptr, ITensor *output = nullptr,
                   int result_fixedpoint_multiplier = 0, int result_shift = 0, int result_offset_after_shift = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDirectConvolutionLayerOutputStageKernel
     *
     * @param[in] input  Input to add the bias to. If @p output is not specified then accumulation is done in-place.
     *                   Data type supported: QS16/QS32/F16/F32
     * @param[in] bias   (Optional) The shared bias tensor to add. It must be 1D Tensor. Data type supported: Same as @p input
     * @param[in] output (Optional) If the output tensor is specified the accumulation is done out-of-place. (Defaults to nullptr)
     *                         Data type supported: QS8/QS16/F16/F32
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias = nullptr, const ITensorInfo *output = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using OutputStageKernel = void(ITensor *input, const ITensor *bias, const Window &window, ITensor *output,
                                   int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift);

private:
    OutputStageKernel *_func;
    ITensor           *_input;
    const ITensor     *_bias;
    ITensor           *_output;
    int                _result_fixedpoint_multiplier;
    int                _result_shift;
    int                _result_offset_after_shift;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYEROUTPUTSTAGEKERNEL_H__ */
