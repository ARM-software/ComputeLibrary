/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSCALE_H__
#define __ARM_COMPUTE_CLSCALE_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLScaleKernel */
class CLScale : public ICLSimpleFunction
{
public:
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8/QASYMM8/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor. Data types supported: Same as @p input
     *                                      All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]     policy                The interpolation type.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     * @param[in]     sampling_policy       (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     */
    void configure(ICLTensor *input, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value = PixelValue(),
                   SamplingPolicy sampling_policy = SamplingPolicy::CENTER);

    /** Static function to check if given info will lead to a valid configuration of @ref CLScale
     *
     * @param[in] input                 Source tensor info. Data types supported: U8/QASYMM8/S16/F16/F32.
     * @param[in] output                Output tensor info. Data type supported: Same as @p input
     *                                  All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in] policy                The interpolation type.
     * @param[in] border_mode           Strategy to use for borders.
     * @param[in] constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     * @param[in] sampling_policy       (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value = PixelValue(),
                           SamplingPolicy sampling_policy = SamplingPolicy::CENTER);
};
}
#endif /*__ARM_COMPUTE_CLSCALE_H__ */
