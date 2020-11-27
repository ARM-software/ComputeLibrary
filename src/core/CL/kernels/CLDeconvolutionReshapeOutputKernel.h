/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLDECONVOLUTIONLAYERRESHAPEOUTPUTKERNEL_H
#define ARM_COMPUTE_CLDECONVOLUTIONLAYERRESHAPEOUTPUTKERNEL_H

#include "src/core/CL/ICLSimpleKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the OpenCL kernel to be used for reshaping the tensor before returning the result of deconvolution.
 *
 * The input tensor to this OpenCL kernel is expected to be the result of a @ref CLGEMM operation between the Deconvolution input and the Deconvolution filter.
 *
 * The input tensor should have the following shape: [filter_width * filter_height * ofms, width, height, batch_size]
 *
 * The output tensor should have the following shape: [stride_x * (input_width - 1) + filter_width - 2 * padx, stride_y * (input_height - 1) + filter_height - 2 * pady, ofms, batch_size]
 *
 * For example, given a tensor with dimensions [4, 2, 2] this function returns a tensor with dimensions [1, 4, 4].
 *
 */
class CLDeconvolutionReshapeOutputKernel : public ICLSimpleKernel
{
public:
    /** Default constructor */
    CLDeconvolutionReshapeOutputKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionReshapeOutputKernel(const CLDeconvolutionReshapeOutputKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionReshapeOutputKernel &operator=(const CLDeconvolutionReshapeOutputKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLDeconvolutionReshapeOutputKernel(CLDeconvolutionReshapeOutputKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLDeconvolutionReshapeOutputKernel &operator=(CLDeconvolutionReshapeOutputKernel &&) = default;
    /** Default destructor */
    ~CLDeconvolutionReshapeOutputKernel() = default;

    /** Initialise the kernel's source and destination.
     *
     * @param[in]  input        Input tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in]  bias         Bias tensor to be added directly during the reshape operation. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[out] output       Output tensor with the following shape: [stride_x * (input_width - 1) + filter_width - 2 * padx, stride_y * (input_height - 1) + filter_height - 2 * pady, ofms, batch_size]
     *                          Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in]  input_info   Deconvolution input tensor info. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in]  weights_info Deconvolution weights tensor info. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in]  deconv_info  Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo. This kernel supports only stride_x = weights.width && stride_y = weights.height. Moreover, padding is not supported.
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const ITensorInfo *input_info, const ITensorInfo *weights_info, const PadStrideInfo &deconv_info);
    /** Initialise the kernel's source and destination.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in]  bias            Bias tensor to be added directly during the reshape operation. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[out] output          Output tensor with the following shape: [stride_x * (input_width - 1) + filter_width - 2 * padx, stride_y * (input_height - 1) + filter_height - 2 * pady, ofms, batch_size]
     *                             Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in]  input_info      Deconvolution input tensor info. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in]  weights_info    Deconvolution weights tensor info. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in]  deconv_info     Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo. This kernel supports only stride_x = weights.width && stride_y = weights.height. Moreover, padding is not supported.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const ITensorInfo *input_info, const ITensorInfo *weights_info,
                   const PadStrideInfo &deconv_info);

    /** Static function to check if given info will lead to a valid configuration of @ref  CLDeconvolutionReshapeOutputKernel.
     *
     * @param[in] input        GEMM output tensor info to be reshaped. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in] bias         (Optional) Optional bias tensor info to be added directly during the reshape operation. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in] output       Reshaped output tensor info. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in] input_info   Original input tensor info. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in] weights_info Original weights tensor info output. Supported data types: same as @p input.  Supported data layouts: same as @p input.
     * @param[in] deconv_info  Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo. This kernel supports only stride_x = weights.width && stride_y = weights.height. Moreover, padding is not supported.
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const ITensorInfo *input_info, const ITensorInfo *weights_info, const PadStrideInfo &deconv_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    bool             _add_bias;
    const ICLTensor *_bias;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDECONVOLUTIONLAYERRESHAPEOUTPUTKERNEL_H */
