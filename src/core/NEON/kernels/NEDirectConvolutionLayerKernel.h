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
#ifndef ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERKERNEL_H
#define ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon interface for Direct Convolution Layer kernel */
class NEDirectConvolutionLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDirectConvolutionLayerKernel";
    }
    /** Default constructor */
    NEDirectConvolutionLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerKernel(const NEDirectConvolutionLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerKernel &operator=(const NEDirectConvolutionLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerKernel(NEDirectConvolutionLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerKernel &operator=(NEDirectConvolutionLayerKernel &&) = default;
    /** Default destructor */
    ~NEDirectConvolutionLayerKernel() = default;
    /** Set the input, weights, and output tensors.
     *
     * @note: DirectConvolution only works in the following configurations:
     *        1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3
     *        3x3 convolution with stride_x = 1/2/3, stride_y = 1/2/3
     *
     * @param[in]  input     The input tensor to convolve. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F16/F32.
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                       The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                       Data type supported:Same as @p input.
     * @param[out] output    Output tensor.
     *                       The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: F16/F32
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDirectConvolutionLayerKernel
     *
     * @param[in] input     The input tensor to convolve. 3 lower dimensions represent a single input [width, height, IFM],
     *                      while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F16/F32.
     * @param[in] weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                      The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                      Data type supported:Same as @p input.
     * @param[in] output    Output tensor.
     *                      The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: F16/F32
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /* Template function for optimized convolution NHWC */
    template <typename T>
    void convolve_nhwc_optimized(const Window &window);

    /* Template function for convolution NHWC */
    template <typename T>
    void convolve_nhwc(const Window &window);

    const ITensor *_input;
    const ITensor *_weights;
    ITensor       *_output;
    PadStrideInfo  _conv_info;
    BorderSize     _border_size;
    unsigned int   _kernel_size;
    unsigned int   _num_weight_elems_read_per_row;
    unsigned int   _num_elems_read_per_iteration;
    unsigned int   _num_elems_written_per_iteration;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERKERNEL_H */
