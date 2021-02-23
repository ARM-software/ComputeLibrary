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
#ifndef ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYEROUTPUTSTAGEKERNEL_H
#define ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYEROUTPUTSTAGEKERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;
/** Neon kernel to accumulate the biases, if provided, or downscale in case of quantized input.
 *
 * @note We assume bias to be shared
 * @note For quantized computations (i.e. @p input of S32 type) the output data type for auto-initialization must be passed as part
 *       of the @ref DirectConvolutionLayerOutputStageKernelInfo.
 */
class NEDirectConvolutionLayerOutputStageKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDirectConvolutionLayerOutputStageKernel";
    }
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
     * @param[in, out] input  Input to add the bias to. If @p output is not specified then accumulation is done in-place.
     *                        Data type supported: F16/F32/S32
     * @param[in]      bias   (Optional) The shared bias tensor to add. It must be 1D Tensor. Data type supported: Same as @p input
     * @param[out]     output (Optional) If the output tensor is specified the accumulation is done out-of-place. (Defaults to nullptr)
     *                        Note that in-place computation is only supported for F16/F32. For S32 this must not be nullptr.
     *                        Data type supported: F16/F32 or QASYMM8/QASYMM8_SIGNED if @p input is S32
     * @param[in]      info   (Optional) DirectConvolutionLayerOutputStageKernel descriptor metadata
     */
    void configure(ITensor *input, const ITensor *bias = nullptr, ITensor *output = nullptr,
                   const DirectConvolutionLayerOutputStageKernelInfo &info = DirectConvolutionLayerOutputStageKernelInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEDirectConvolutionLayerOutputStageKernel
     *
     * @param[in] input  Input to add the bias to. If @p output is not specified then accumulation is done in-place.
     *                   Data type supported: F16/F32/S32
     * @param[in] bias   (Optional) The shared bias tensor to add. It must be 1D Tensor. Data type supported: Same as @p input
     * @param[in] output (Optional) If the output tensor is specified the accumulation is done out-of-place. (Defaults to nullptr)
     *                   Note that in-place computation is only supported for F16/F32. For S32 this must not be nullptr.
     *                   Data type supported: F16/F32 or QASYMM8/QASYMM8_SIGNED if @p input is S32
     * @param[in] info   (Optional) DirectConvolutionLayerOutputStageKernel descriptor metadata
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias = nullptr, const ITensorInfo *output = nullptr,
                           const DirectConvolutionLayerOutputStageKernelInfo &info = DirectConvolutionLayerOutputStageKernelInfo());

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using OutputStageKernel = void(ITensor *input, const ITensor *bias, const Window &window, ITensor *output,
                                   int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift, bool has_bias);

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
#endif /*ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYEROUTPUTSTAGEKERNEL_H */
