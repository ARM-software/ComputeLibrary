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
#ifndef __ARM_COMPUTE_CLSCALEKERNEL_H__
#define __ARM_COMPUTE_CLSCALEKERNEL_H__

#include "arm_compute/core/CL/ICLSimple2DKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the scale kernel */
class CLScaleKernel : public ICLSimple2DKernel
{
public:
    /** Initialise the kernel's inputs, output and interpolation policy
     *
     * @param[in]  input           Source tensor. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[out] output          Destination tensor. Data types supported: Same as @p input
     *                             All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]  policy          Interpolation type to use
     * @param[in]  border_mode     Selected border mode.
     * @param[in]  sampling_policy (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     */
    void configure(const ICLTensor *input, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy = SamplingPolicy::CENTER);

    /** Static function to check if given info will lead to a valid configuration of @ref CLScaleKernel
     *
     * @param[in] input           Source tensor info. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[in] output          Destination tensor info. Data types supported: Same as @p input
     *                            All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in] policy          Interpolation type to use
     * @param[in] border_mode     Selected border mode.
     * @param[in] sampling_policy (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy = SamplingPolicy::CENTER);
    /** Input tensor accessor.
     *
     * @return Pointer to input tensor.
     */
    const ICLTensor *input() const;
    /** Output tensor accessor.
     *
     * @return Pointer to output tensor.
     */
    const ICLTensor *output() const;

    // Inherited methods overridden:
    BorderSize border_size() const override;
    void run(const Window &window, cl::CommandQueue &queue) override;

public:
    InterpolationPolicy _interpolationPolicy = InterpolationPolicy::BILINEAR;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLSCALEKERNEL_H__ */
