/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTS_H__
#define __ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTS_H__

#include "arm_compute/core/CL/kernels/CLConvertFullyConnectedWeightsKernel.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLConvertFullyConnectedWeightsKernel. */
class CLConvertFullyConnectedWeights : public ICLSimpleFunction
{
public:
    /** Initialize the function.
     *
     * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
     * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in]  data_layout          The data layout the weights have been trained in.
     *
     * @return A status
     */
    void configure(const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvertFullyConnectedWeights
     *
     * @param[in] input                Source weights tensor info to convert. Must be 2 dimensional. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in] output               The converted weights tensor info. Shape and Data Type: Same as @p input.
     * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in] data_layout          The data layout the weights have been trained in.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape, DataLayout data_layout);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTS_H__ */
