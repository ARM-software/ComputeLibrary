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
#ifndef ARM_COMPUTE_CLDIRECTCONVOLUTIONLAYERKERNEL_H
#define ARM_COMPUTE_CLDIRECTCONVOLUTIONLAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the  direct convolution kernel.
 */
class CLDirectConvolutionLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDirectConvolutionLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDirectConvolutionLayerKernel(const CLDirectConvolutionLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDirectConvolutionLayerKernel &operator=(const CLDirectConvolutionLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLDirectConvolutionLayerKernel(CLDirectConvolutionLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLDirectConvolutionLayerKernel &operator=(CLDirectConvolutionLayerKernel &&) = default;
    /** Default destructor */
    ~CLDirectConvolutionLayerKernel() = default;
    /** Set the input, weights, biases and output tensors.
     *
     * @note: DirectConvolution only works in the following configurations:
     *        1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3
     *        3x3 convolution with stride_x = 1/2, stride_y = 1/2
     *        5x5 convolution with stride_x = 1/2, stride_y = 1/2
     *        9x9 convolution with stride_x = 1/2, stride_y = 1/2
     *
     * @param[in]  input     The input tensor to convolve. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                       The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                       Data type supported:Same as @p input.
     * @param[in]  biases    Biases tensor. Biases are 1D tensor with dimension [OFM].
     *                       Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type
     * @param[out] output    Output tensor.
     *                       The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info);
    /** Set the input, weights, biases and output tensors.
     *
     * @note: DirectConvolution only works in the following configurations:
     *        1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3
     *        3x3 convolution with stride_x = 1/2, stride_y = 1/2
     *        5x5 convolution with stride_x = 1/2, stride_y = 1/2
     *        9x9 convolution with stride_x = 1/2, stride_y = 1/2
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           The input tensor to convolve. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in]  weights         Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                             Data type supported:Same as @p input.
     * @param[in]  biases          Biases tensor. Biases are 1D tensor with dimension [OFM].
     *                             Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type
     * @param[out] output          Output tensor.
     *                             The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDirectConvolutionLayerKernel
     *
     * @param[in] input     The input tensor to convolve. 3 lower dimensions represent a single input [width, height, IFM],
     *                      while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[in] weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                      The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                      Data type supported:Same as @p input.
     * @param[in] biases    Biases tensor. Biases are 1D tensor with dimension [OFM].
     *                      Data type supported: Should match @p input data type, except for input of QASYMM8 and QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] output    Output tensor.
     *                      The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] target    Target GPU architecture.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, const GPUTarget target);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

public:
    const ICLTensor *_input;
    const ICLTensor *_biases;
    const ICLTensor *_weights;
    ICLTensor       *_output;
    DataLayout       _data_layout;
    BorderSize       _border_size;
    int              _conv_stride_x;
    int              _conv_stride_y;
    PadStrideInfo    _conv_info;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDIRECTCONVOLUTIONLAYERKERNEL_H */
