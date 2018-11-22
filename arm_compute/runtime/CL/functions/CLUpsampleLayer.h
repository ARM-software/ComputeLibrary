/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLUPSAMPLELAYER_H__
#define __ARM_COMPUTE_CLUPSAMPLELAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLUpsampleLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLUpsampleLayerKernel */
class CLUpsampleLayer : public IFunction
{
public:
    /** Default constructor */
    CLUpsampleLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLUpsampleLayer(const CLUpsampleLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLUpsampleLayer &operator=(const CLUpsampleLayer &) = delete;
    /** Allow instances of this class to be moved */
    CLUpsampleLayer(CLUpsampleLayer &&) = default;
    /** Allow instances of this class to be moved */
    CLUpsampleLayer &operator=(CLUpsampleLayer &&) = default;
    /** Default destructor */
    virtual ~CLUpsampleLayer() = default;

    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in]  input             Source tensor. Data type supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output            Destination tensor. Data types supported: same as @p input.
     * @param[in]  info              Contains stride information described in @ref Size2D.
     * @param[in]  upsampling_policy Defines the policy to fill the intermediate pixels.
     */
    void configure(ICLTensor *input, ICLTensor *output,
                   const Size2D &info, const InterpolationPolicy upsampling_policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayerUpsample
     *
     * @param[in] input             Source tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] output            Destination tensor info. Data types supported: same as @p input.
     * @param[in] info              Contains  stride information described in @ref Size2D.
     * @param[in] upsampling_policy Defines the policy to fill the intermediate pixels.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                           const Size2D &info, const InterpolationPolicy upsampling_policy);

    // Inherited methods overridden:
    void run() override;

private:
    CLUpsampleLayerKernel _upsample;
    ICLTensor            *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLUPSAMPLELAYER_H__ */
