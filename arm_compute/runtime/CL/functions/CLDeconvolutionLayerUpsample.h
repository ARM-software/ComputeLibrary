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
#ifndef __ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLE_H__
#define __ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLE_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLDeconvolutionLayerUpsampleKernel.h"
#include "arm_compute/core/CL/kernels/CLMemsetKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to execute deconvolution upsample on OpenCL. This function calls the following OpenCL kernels and functions:
 *
 * -# @ref CLMemsetKernel
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
    virtual ~CLDeconvolutionLayerUpsample() = default;

    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in, out] input        Source tensor. Data type supported: QASYMM8/F16/F32.
     * @param[out]     output       Destination tensor. Data type supported: same as @p input.
     * @param[in]      inner_border The number of zeros added to right and top edges of the input.
     * @param[in]      info         Contains padding and policies to be used in the deconvolution.
     */
    void configure(ICLTensor *input, ICLTensor *output, const BorderSize &inner_border,
                   const PadStrideInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayerUpsample
     *
     * @param[in] input        Source tensor info. Data type supported: QASYMM8/F16/F32.
     * @param[in] output       Destination tensor info. Data type supported: same as @p input.
     * @param[in] inner_border The number of zeros added to right and top edges of the input.
     * @param[in] info         Contains padding and policies to be used in the deconvolution.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const BorderSize &inner_border,
                           const PadStrideInfo &info);

    // Inherited methods overridden:
    void run() override;

private:
    CLDeconvolutionLayerUpsampleKernel _upsample;
    CLMemsetKernel                     _memset;
    ICLTensor                         *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLDECONVOLUTIONLAYERUPSAMPLE_H__ */
