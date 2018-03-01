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
#ifndef __ARM_COMPUTE_CLWINOGRADINPUTTRANSFORM_H__
#define __ARM_COMPUTE_CLWINOGRADINPUTTRANSFORM_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute a @ref CLWinogradInputTransformKernel. */
class CLWinogradInputTransform : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * @param[in] input       The input tensor to transform. Data types supported: F32
     * @param[in] output      The output tensor. Data types supported: Same as @p input
     * @param[in] conv_info   Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     * @param[in] kernel_dims Kernel dimensions. Currently only 3x3 kernels are supported
     */
    void configure(ICLTensor *input, ICLTensor *output, const PadStrideInfo &conv_info, const Size2D &kernel_dims);
    /**  Static function to check if given info will lead to a valid configuration of @ref CLWinogradInputTransform.
     *
     * @param[in] input       First tensor input info. Data types supported: F32.
     * @param[in] output      Output tensor info. Data types supported: same as @p input.
     * @param[in] conv_info   Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     * @param[in] kernel_dims Kernel dimensions. Currently only 3x3 kernels are supported
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PadStrideInfo &conv_info, const Size2D &kernel_dims);
};
}
#endif /*__ARM_COMPUTE_CLWINOGRADINPUTTRANSFORM_H__ */
