/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEQUANTIZATIONLAYER_H__
#define __ARM_COMPUTE_NEQUANTIZATIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEMinMaxLocationKernel.h"
#include "arm_compute/core/NEON/kernels/NEQuantizationLayerKernel.h"
#include "arm_compute/runtime/Tensor.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Basic function to simulate a quantization layer. This function calls the following NEON kernels:
 *
 * -# @ref NEMinMaxKernel
 * -# @ref NEQuantizationLayerKernel
 *
 */
class NEQuantizationLayer : public IFunction
{
public:
    /** Default constructor */
    NEQuantizationLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: F32
     * @param[out] output Destination tensor. Data types supported: F32
     */
    void configure(const ITensor *input, ITensor *output);

    // Inherited methods overridden:
    void run() override;

private:
    NEQuantizationLayerKernel _quantize_kernel;
    NEMinMaxKernel            _min_max_kernel;
    float                     _min;
    float                     _max;
};
}
#endif /* __ARM_COMPUTE_NEQUANTIZATIONLAYER_H__ */
