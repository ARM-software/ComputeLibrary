/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NESCALEIMAGE_H
#define ARM_COMPUTE_NESCALEIMAGE_H

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEScaleKernel */
class NEScale : public INESimpleFunctionNoBorder
{
public:
    /** Constructor
     *
     * Initialize NEScale
     */
    NEScale();
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in, out] input  Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/U8/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output Destination tensor. Data type supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]      info   @ref ScaleKernelInfo to be used for configuration
     */
    void configure(ITensor *input, ITensor *output, const ScaleKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEScale
     *
     * @param[in] input  Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/U8/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[in] output Destination tensor. Data type supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in] info   @ref ScaleKernelInfo to be used for validation
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ScaleKernelInfo &info);

private:
    Tensor _offsets; /**< Offset to access the element with NEAREST interpolation or the top-left element with BILINEAR interpolation in the input tensor */
    Tensor _dx;      /**< Element's distance between the X real coordinate and the smallest X following integer */
    Tensor _dy;      /**< Element's distance between the Y real coordinate and the smallest Y following integer */
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NESCALEIMAGE_H */
