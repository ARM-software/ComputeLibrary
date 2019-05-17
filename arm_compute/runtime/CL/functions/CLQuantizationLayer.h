/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLQUANTIZATIONLAYER_H__
#define __ARM_COMPUTE_CLQUANTIZATIONLAYER_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to simulate a quantization layer. This function calls the following CL kernels:
 *
 * @note The implementation supports only 3D input tensors.
 *
 * -# @ref CLQuantizationLayerKernel
 *
 */
class CLQuantizationLayer : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. The dimensions over the third will be interpreted as batches. Data types supported: F16/32.
     * @param[out] output Destination tensor with the same dimensions of input. Output data type must be QASYMM8.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLQuantizationLayer
     *
     * @param[in] input  Input tensor info. The dimensions over the third will be interpreted as batches. Data types supported: F16/32.
     * @param[in] output Output tensor info. Output data type must be QASYMM8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};
} //namespace arm_compute
#endif /* __ARM_COMPUTE_CLQUANTIZATIONLAYER_H__ */
