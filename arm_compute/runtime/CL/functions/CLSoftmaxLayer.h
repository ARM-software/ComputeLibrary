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
#ifndef ARM_COMPUTE_CLSOFTMAXLAYER_H
#define ARM_COMPUTE_CLSOFTMAXLAYER_H

#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLFlattenLayer.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to compute a SoftmaxLayer.
 *
 * Softmax is calculated by :
 * @f[ out = exp((x - max(x)) * beta) / sum(exp((x - max(x)) * beta)) @f]
 *
 * Log Softmax is calculated by :
 * @f[ out = (x - max(x) * beta) - \sum{e^{x - max(x) * beta}} @f]
 *
 * This function runs the following kernels:
 * -# @ref CLLogits1DMaxKernel
 * -# @ref CLLogits1DShiftExpSumKernel
 * -# @ref CLLogits1DNormKernel
 * And if the reduce_end_axis is not 0, the function will use one of the the following kernels to reshape the input and
 * perform softmax on the reshaped input:
 * -# @ref CLFlattenLayerKernel
 * -# @ref CLReshapeLayerKernel
 */
template <bool IS_LOG = false>
class CLSoftmaxLayerGeneric : public IFunction
{
public:
    /** Constructor */
    CLSoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/F16/F32
     * @param[out] output          Destination tensor. Data types supported: same as @p input
     * @param[in]  beta            (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in]  reduce_end_axis (Optional) The last axis of the first n dimensions (inclusive)to reduce. Defaults to 0.
     *                   It has the purpose of squashing together the first n dimensions till (including) the @p reduce_end_axis. For instance, given a [2x3x4x5] image,
     *                   when @p reduce_end_axis is 1, the reduction will be applied to axes 0 and 1, and the Softmax op will be applied on each of the [2x3] planes of the input image.
     *                   Must be in range [0, input_num_dimensions).
     */
    void configure(const ICLTensor *input, ICLTensor *output, float beta = 1.0f, size_t reduce_end_axis = 0);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/F16/F32
     * @param[out] output          Destination tensor. Data types supported: same as @p input
     * @param[in]  beta            (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in]  reduce_end_axis (Optional) The last axis of the first n dimensions (inclusive)to reduce. Defaults to 0.
     *                   It has the purpose of squashing together the first n dimensions till (including) the @p reduce_end_axis. For instance, given a [2x3x4x5] image,
     *                   when @p reduce_end_axis is 1, the reduction will be applied to axes 0 and 1, and the Softmax op will be applied on each of the [2x3] planes of the input image.
     *                   Must be in range [0, input_num_dimensions).
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, float beta = 1.0f, size_t reduce_end_axis = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSoftmaxLayer
     *
     * @param[in] input           Source tensor. Data types supported: QASYMM8/F16/F32
     * @param[in] output          Destination tensor. Data types supported: same as @p input
     * @param[in] beta            (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in] reduce_end_axis (Optional) The last axis of the first n dimensions (inclusive)to reduce. Defaults to 0.
     *                   It has the purpose of squashing together the first n dimensions till (including) the @p reduce_end_axis. For instance, given a [2x3x4x5] image,
     *                   when @p reduce_end_axis is 1, the reduction will be applied to axes 0 and 1, and the Softmax op will be applied on each of the [2x3] planes of the input image.
     *                   Must be in range [0, input_num_dimensions).
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, float beta = 1.0f, size_t reduce_end_axis = 0);

    // Inherited methods overridden:
    void run() override;

private:
    /** Utility method to configure the kernels needed to flatten the input
     * tensor.
     *
     * @note This function changes the internal state of this class. In particular,
     * it initializes the kernel @p _flatten_kernel and the tensors @p _input_flat and
     * @p _output_flat
     *
     * @param[in] input           Original source tensor.
     * @param[in] output          Original destination tensor.
     * @param[in] reduce_end_axis (Optional) The last axis of the first n dimensions (inclusive)to reduce. Defaults to 0.
     *                   It has the purpose of squashing together the first n dimensions till (including) the @p reduce_end_axis. For instance, given a [2x3x4x5] image,
     *                   when @p reduce_end_axis is 1, the reduction will be applied to axes 0 and 1, and the Softmax op will be applied on each of the [2x3] planes of the input image.
     *                   Must be in range [0, input_num_dimensions).
     */
    void configure_reshape_input_kernel(const ICLTensor *input, const ICLTensor *output, size_t reduce_end_axis);
    /** Utility method to configure the kernels needed to flatten the input
     * tensor.
     *
     * @note This function changes the internal state of this class. In particular,
     * it initializes the kernel @p _flatten_kernel and the tensors @p _input_flat and
     * @p _output_flat
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] input           Original source tensor.
     * @param[in] output          Original destination tensor.
     * @param[in] reduce_end_axis (Optional) The last axis of the first n dimensions (inclusive)to reduce. Defaults to 0.
     *                   It has the purpose of squashing together the first n dimensions till (including) the @p reduce_end_axis. For instance, given a [2x3x4x5] image,
     *                   when @p reduce_end_axis is 1, the reduction will be applied to axes 0 and 1, and the Softmax op will be applied on each of the [2x3] planes of the input image.
     *                   Must be in range [0, input_num_dimensions).
     */
    void configure_reshape_input_kernel(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *output, size_t reduce_end_axis);

    MemoryGroup                    _memory_group;
    CLLogits1DMaxShiftExpSumKernel _max_shift_exp_sum_kernel;
    CLLogits1DNormKernel           _norm_kernel;
    std::unique_ptr<IFunction>     _flatten_ptr;
    CLReshapeLayer                 _reshape;
    CLTensor                       _max;
    CLTensor                       _sum;
    CLTensor                       _tmp;
    CLTensor                       _input_flattened;
    CLTensor                       _output_flattened;
    bool                           _needs_flattening;
};

using CLSoftmaxLayer    = CLSoftmaxLayerGeneric<false>;
using CLLogSoftmaxLayer = CLSoftmaxLayerGeneric<true>;
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLSOFTMAXLAYER_H */
