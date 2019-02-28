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
#ifndef __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERRESHAPEWEIGHTSGENERICKERNEL_H__
#define __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERRESHAPEWEIGHTSGENERICKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the depthwise weights reshape kernel.
 *  This kernel reshape original weights' low 2D dimensions into a single row and
 *  have the second dimension as the original depth size.
 *
 **/
class CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel(const CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel &operator=(const CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel(CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel &operator=(CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM].
     *                    Data type supported: QASYMM8/F16/F32.
     * @param[out] output The output tensor. Data type supported: same as @p input.
     * @param[in]  biases (Optional) The input biases to add. Shape [IFM]. Data type supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *biases = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel
     *
     * @param[in] input  The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM].
     *                   Data type supported: QASYMM8/F32.
     * @param[in] output The output tensor. Data type supported: same as @p input.
     * @param[in] biases (Optional) The input biases to add. Shape [IFM]. Data type supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *biases = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_biases;
    ICLTensor       *_output;
};
} // arm_compute
#endif /*__ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERRESHAPEWEIGHTSGENERICKERNEL_H__ */
