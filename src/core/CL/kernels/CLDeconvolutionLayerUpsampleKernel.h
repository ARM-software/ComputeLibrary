/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLEKERNEL_H
#define ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLEKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the Deconvolution layer kernel on OpenCL.
 */
class CLDeconvolutionLayerUpsampleKernel : public ICLKernel
{
public:
    /** Constructor */
    CLDeconvolutionLayerUpsampleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionLayerUpsampleKernel(const CLDeconvolutionLayerUpsampleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionLayerUpsampleKernel &operator=(const CLDeconvolutionLayerUpsampleKernel &) = delete;
    /** Default Move Constructor. */
    CLDeconvolutionLayerUpsampleKernel(CLDeconvolutionLayerUpsampleKernel &&) = default;
    /** Default move assignment operator */
    CLDeconvolutionLayerUpsampleKernel &operator=(CLDeconvolutionLayerUpsampleKernel &&) = default;
    /** Default destructor */
    ~CLDeconvolutionLayerUpsampleKernel() = default;

    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Source tensor. Data types supported: All.
     * @param[out] output Destination tensor. Data types supported: same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]  info   Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const PadStrideInfo &info);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: All.
     * @param[out] output          Destination tensor. Data types supported: same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]  info            Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const PadStrideInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayerUpsample
     *
     * @param[in] input  Source tensor info. Data types supported: All.
     * @param[in] output Destination tensor info. Data types supported: same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in] info   Contains padding and stride information described in @ref PadStrideInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PadStrideInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    PadStrideInfo    _info;
    DataLayout       _data_layout;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLEKERNEL_H */
