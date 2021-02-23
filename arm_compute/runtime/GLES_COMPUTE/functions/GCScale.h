/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GCSCALE_H
#define ARM_COMPUTE_GCSCALE_H

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class IGCTensor;

/** Basic function to run @ref GCScaleKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCScale : public IGCSimpleFunction
{
public:
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in,out] input  Source tensor. Data types supported: F16. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output Destination tensor. Data types supported: Same as @p input
     *                       All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]     info   @ref ScaleKernelInfo descriptor to be used to configure
     */
    void configure(IGCTensor *input, IGCTensor *output, const ScaleKernelInfo &info);
};
}
#endif /*ARM_COMPUTE_GCSCALE_H */
