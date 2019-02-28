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
#ifndef __ARM_COMPUTE_NESCALEIMAGE_H__
#define __ARM_COMPUTE_NESCALEIMAGE_H__

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEScaleKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEScaleKernel */
class NEScale : public IFunction
{
public:
    /** Constructor
     *
     * Initialize NEScale
     */
    NEScale();
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Destination tensor. Data type supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]      policy                The interpolation type.
     * @param[in]      border_mode           Strategy to use for borders.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     * @param[in]      sampling_policy       (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     */
    void configure(ITensor *input, ITensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value = PixelValue(),
                   SamplingPolicy sampling_policy = SamplingPolicy::CENTER);
    /** Static function to check if given info will lead to a valid configuration of @ref NEScale
     *
     * @param[in] input                 Source tensor. Data type supported: U8/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[in] output                Destination tensor. Data type supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in] policy                The interpolation type.
     * @param[in] border_mode           Strategy to use for borders.
     * @param[in] constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     * @param[in] sampling_policy       (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy, BorderMode border_mode,
                           PixelValue constant_border_value = PixelValue(), SamplingPolicy sampling_policy = SamplingPolicy::CENTER);

    // Inherited methods overridden:
    void run() override;

private:
    Tensor             _offsets;        /**< Offset to access the element with NEAREST interpolation or the top-left element with BILINEAR interpolation in the input tensor */
    Tensor             _dx;             /**< Element's distance between the X real coordinate and the smallest X following integer */
    Tensor             _dy;             /**< Element's distance between the Y real coordinate and the smallest Y following integer */
    NEScaleKernel      _scale_kernel;   /**< Kernel to perform the scaling */
    NEFillBorderKernel _border_handler; /**< kernel to handle tensor borders */
};
}
#endif /*__ARM_COMPUTE_NESCALEIMAGE_H__ */
