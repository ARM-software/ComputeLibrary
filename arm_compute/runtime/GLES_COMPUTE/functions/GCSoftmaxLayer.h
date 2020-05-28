/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_GCSOFTMAXLAYER_H
#define ARM_COMPUTE_GCSOFTMAXLAYER_H

#include "arm_compute/core/GLES_COMPUTE/kernels/GCSoftmaxLayerKernel.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to compute a SoftmaxLayer.
 *
 * Softmax is calculated by :
 * @f[ out = exp(x - max(x)) / sum(exp(x - max(x))) @f]
 *
 * This function runs the following kernels:
 * -# @ref GCLogits1DMaxKernel
 * -# @ref GCLogits1DShiftExpSumKernel
 * -# @ref GCLogits1DNormKernel
 */
class GCSoftmaxLayer : public IFunction
{
public:
    /** Constructor */
    GCSoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input           Source tensor. Data types supported: F16/F32
     * @param[out] output          Destination tensor. Data types supported: same as @p input
     * @param[in]  beta            (Optional) A scaling factor for the exponent. Only beta = 1 is supported
     * @param[in]  reduce_end_axis (Optional) The last axis of the first n dimensions (inclusive)to reduce. Defaults to 0.
     *                   It has the purpose of squashing together the first n dimensions till (including) the @p reduce_end_axis. For instance, given a [2x3x4x5] image,
     *                   when @p reduce_end_axis is 1, the reduction will be applied to axes 0 and 1, and the Softmax op will be applied on each of the [2x3] planes of the input image.
     *                   Must be in range [0, input_num_dimensions).
     *
     * @note The value of @p reduce_end_axis must be always 0 for GLES
     */
    void configure(const IGCTensor *input, IGCTensor *output, float beta = 1.0f, size_t reduce_end_axis = 0);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                 _memory_group;
    GCLogits1DMaxKernel         _max_kernel;
    GCLogits1DShiftExpSumKernel _shift_exp_sum_kernel;
    GCLogits1DNormKernel        _norm_kernel;
    GCTensor                    _max;
    GCTensor                    _sum;
    GCTensor                    _tmp;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_GCSOFTMAXLAYER_H */
