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
#ifndef ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLE_H
#define ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLFill.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLDeconvolutionLayerUpsampleKernel;
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to execute deconvolution upsample on OpenCL. This function calls the following OpenCL kernels and functions:
 *
 * -# @ref CLFill
 * -# @ref CLDeconvolutionLayerUpsampleKernel
 */
class CLDeconvolutionLayerUpsample : public IFunction
{
public:
    /** Default constructor */
    CLDeconvolutionLayerUpsample();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionLayerUpsample(const CLDeconvolutionLayerUpsample &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDeconvolutionLayerUpsample &operator=(const CLDeconvolutionLayerUpsample &) = delete;
    /** Allow instances of this class to be moved */
    CLDeconvolutionLayerUpsample(CLDeconvolutionLayerUpsample &&) = default;
    /** Allow instances of this class to be moved */
    CLDeconvolutionLayerUpsample &operator=(CLDeconvolutionLayerUpsample &&) = default;
    /** Default destructor */
    ~CLDeconvolutionLayerUpsample();

    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in, out] input  Source tensor. Data type supported: All.
     * @param[out]     output Destination tensor. Data type supported: same as @p input.
     * @param[in]      info   Contains padding and policies to be used in the deconvolution.
     */
    void configure(ICLTensor *input, ICLTensor *output, const PadStrideInfo &info);
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input           Source tensor. Data type supported: All.
     * @param[out]     output          Destination tensor. Data type supported: same as @p input.
     * @param[in]      info            Contains padding and policies to be used in the deconvolution.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const PadStrideInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayerUpsample
     *
     * @param[in] input  Source tensor info. Data type supported: All.
     * @param[in] output Destination tensor info. Data type supported: same as @p input.
     * @param[in] info   Contains padding and policies to be used in the deconvolution.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PadStrideInfo &info);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<CLDeconvolutionLayerUpsampleKernel> _upsample;
    CLFill                                              _fill;
    ICLTensor                                          *_output;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLE_H */
