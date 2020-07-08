/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEUPSAMPLELAYER_H
#define ARM_COMPUTE_NEUPSAMPLELAYER_H

#include "arm_compute/core/NEON/kernels/NEUpsampleLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Function to run upsample layer */
class NEUpsampleLayer : public IFunction
{
public:
    /** Constructor */
    NEUpsampleLayer();
    /** Set the input output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     * @param[in]  info   Contains stride information described in @ref Size2D.
     * @param[in]  policy Defines the policy to fill the intermediate pixels.
     *
     */
    void configure(const ITensor *input, ITensor *output, const Size2D &info,
                   const InterpolationPolicy &policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEUpsampleLayer
     *
     * @param[in]  input  Source tensor info. Data types supported: QASYMM8_SIGNED/QASYMM8/F16/F32.
     * @param[out] output Destination tensor info. Data types supported: same as @p input.
     * @param[in]  info   Contains stride information described in @ref Size2D.
     * @param[in]  policy Defines the policy to fill the intermediate pixels.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &info,
                           const InterpolationPolicy &policy);

    // Inherited methods overridden:
    void run() override;

private:
    NEUpsampleLayerKernel _kernel;
    DataLayout            _data_layout;
};
} // arm_compute
#endif /* ARM_COMPUTE_NEUPSAMPLELAYER_H */
