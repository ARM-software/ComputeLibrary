/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDEPTHWISEIM2COLKERNEL_H__
#define __ARM_COMPUTE_NEDEPTHWISEIM2COLKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
class ITensor;

/** Interface for the depthwise im2col reshape kernel.
 *  This kernel reshape the input low 3 dimensions to a new 3D shape  where the output's first dimension is
 *  the linear patch size (FILTER_WIDTH * FILTER_HEIGHT) and second dimension is number of patches in per image and third dimension unchanged .
 **/
class NEDepthwiseIm2ColKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDepthwiseIm2ColKernel";
    }
    /** Default constructor */
    NEDepthwiseIm2ColKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseIm2ColKernel(const NEDepthwiseIm2ColKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseIm2ColKernel &operator=(const NEDepthwiseIm2ColKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDepthwiseIm2ColKernel(NEDepthwiseIm2ColKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDepthwiseIm2ColKernel &operator=(NEDepthwiseIm2ColKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input            The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     * @param[out] output           The output tensor. First 3 lower dimensions represent a transform of each 3D input,
     *                              while every dimension above 3 represents a batch. Data types supported: Same as @p input
     * @param[in]  kernel_dims      The kernel dimensions (width and height).
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias         Boolean that specifies if the depthwise convolution has bias.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    void configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias = false, unsigned int depth_multiplier = 1,
                   const Size2D &dilation = Size2D(1U, 1U));

    /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseIm2ColKernel
     *
     * @param[in] input            The input tensor info to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     * @param[in] output           The output tensor info. First 3 lower dimensions represent a transform of each 3D input,
     *                             while every dimension above 3 represents a batch. Data types supported: Same as @p input
     * @param[in] kernel_dims      The kernel dimensions (width and height).
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] has_bias         Boolean that specifies if the depthwise convolution has bias.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias = false, unsigned int depth_multiplier = 1,
                           const Size2D &dilation = Size2D(1U, 1U));

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run the im2col used for the depthwise convolution layer case
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T>
    void run_generic(const Window &window);
    /** Common signature for all the specialised depthwise im2col functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using DepthwiseIm2ColFunctionPtr = void (NEDepthwiseIm2ColKernel::*)(const Window &window);

private:
    DepthwiseIm2ColFunctionPtr _func;
    const ITensor             *_input;
    ITensor                   *_output;
    Size2D                     _kernel_dims;
    PadStrideInfo              _conv_info;
    bool                       _has_bias;
    unsigned int               _depth_multiplier;
    Size2D                     _dilation;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEDEPTHWISEIM2COLKERNEL_H__ */
