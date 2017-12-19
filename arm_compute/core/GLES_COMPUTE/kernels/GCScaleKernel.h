/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCSCALEKERNEL_H__
#define __ARM_COMPUTE_GCSCALEKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCSimple2DKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the scale kernel */
class GCScaleKernel : public IGCSimple2DKernel
{
public:
    /** Initialise the kernel's inputs, output and interpolation policy
     *
     * @param[in]  input            Source tensor. Data types supported: F16
     * @param[out] output           Destination tensor. Data types supported: Same as @p input
     *                              All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]  policy           Interpolation type to use
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     * @param[in]  sampling_policy  (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     */
    void configure(const IGCTensor *input, IGCTensor *output, InterpolationPolicy policy, bool border_undefined, SamplingPolicy sampling_policy = SamplingPolicy::CENTER);

    // Inherited methods overridden:
    BorderSize border_size() const override;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_GCSCALEKERNEL_H__ */
