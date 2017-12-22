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

#ifndef __ARM_COMPUTE_GCDROPOUTLAYER_H__
#define __ARM_COMPUTE_GCDROPOUTLAYER_H__

#include "arm_compute/core/GLES_COMPUTE/kernels/GCDropoutLayerKernel.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class IGCTensor;
/** Basic function to do dropout op. This function calls the following kernels:
 *
 *  -# @ref GCDropoutLayerKernel
 */
class GCDropoutLayer : public IFunction
{
public:
    /** Constructor */
    GCDropoutLayer();

    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data type supported: F16/F32.
     * @param[out] mask    Destination tensor. Data type supported: Same as @p input.
     * @param[out] output  Destination tensor. Data type supported: Same as @p input.
     * @param[in]  ratio   Dropout ratio
     * @param[in]  forward Forward or backward propagation
     *
     */
    void configure(const IGCTensor *input, IGCTensor *mask, IGCTensor *output, float ratio, bool forward);

    //Inherited methods override
    void run() override;

private:
    GCDropoutLayerKernel _dropout_kernel;
};
}

#endif /* __ARM_COMPUTE_GCDROPOUTLAYER_H__ */
