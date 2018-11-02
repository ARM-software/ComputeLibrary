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
#ifndef __ARM_COMPUTE_CLREORGLAYER_H__
#define __ARM_COMPUTE_CLREORGLAYER_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

class CLReorgLayer : public ICLSimpleFunction
{
public:
    /** Initialise the function's source and destination.
     *
     * @param[in]  input  Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output Destination tensor with tensor shape:
     *                    [width_input / stride, height_input / stride, channels_input * stride * stride, batch_size]. This means the output has
     *                    the same number of input elements. Data types supported: same as @p input.
     * @param[in]  stride Stride value to use for reorganizing the values in the output tensor.
     *                    It defines the spatial distance between 2 consecutive pixels in the x and y direction
     *
     */
    void configure(ICLTensor *input, ICLTensor *output, int32_t stride);
    /** Static function to check if given info will lead to a valid configuration of @ref CLReorgLayer
     *
     * @param[in] input  Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] output Destination tensor with tensor shape:
     *                   [width_input / stride, height_input / stride, channels_input * stride * stride, batch_size]. This means the output has
     *                   the same number of input elements. Data types supported: same as @p input. Data types supported: same as @p input.
     * @param[in] stride Stride value to use for reorganizing the values in the output tensor
     *                   It defines the spatial distance between 2 consecutive pixels in the x and y direction
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, int32_t stride);
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLREORGLAYER_H__ */
