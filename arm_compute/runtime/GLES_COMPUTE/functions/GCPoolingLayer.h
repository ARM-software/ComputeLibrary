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
#ifndef __ARM_COMPUTE_GCPOOLINGLAYER_H__
#define __ARM_COMPUTE_GCPOOLINGLAYER_H__

#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to simulate a pooling layer with the specified pooling operation. This function calls the following OpenGL ES kernels:
 *
 * -# @ref GCFillBorderKernel (executed if padding size is different from zero)
 * -# @ref GCPoolingLayerKernel
 */
class GCPoolingLayer : public IGCSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * @param[in,out] input     Source tensor. (Written to only when padding != 0) Data types supported: F16/F32.
     * @param[out]    output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]     pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(IGCTensor *input, IGCTensor *output, const PoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref GCPoolingLayer
     *
     * @param[in] input     Source tensor info. Data types supported: F16/F32.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GCPOOLINGLAYER_H__ */
