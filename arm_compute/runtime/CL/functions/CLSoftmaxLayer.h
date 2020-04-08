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

#include "arm_compute/core/CL/kernels/CLFlattenLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLReshapeLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
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
 */
template <bool IS_LOG = false>
class CLSoftmaxLayerGeneric : public IFunction
{
public:
    /** Constructor */
    CLSoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/F16/F32
     * @param[out] output Destination tensor. Data types supported: same as @p input
     * @param[in]  beta   (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in]  axis   (Optional) Reduction axis. It has the purpose of squashing the first @p axis
     *                    dimensions together. For instance, given a [4x4x4x4] image,
     *                    when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     */
    void configure(const ICLTensor *input, ICLTensor *output, float beta = 1.0f, size_t axis = 1);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/F16/F32
     * @param[out] output          Destination tensor. Data types supported: same as @p input
     * @param[in]  beta            (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in]  axis            (Optional) Reduction axis. It has the purpose of squashing the first @p axis
     *                    dimensions together. For instance, given a [4x4x4x4] image,
     *                    when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, float beta = 1.0f, size_t axis = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSoftmaxLayer
     *
     * @param[in] input  Source tensor. Data types supported: QASYMM8/F16/F32
     * @param[in] output Destination tensor. Data types supported: same as @p input
     * @param[in] beta   (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in] axis   (Optional) Reduction axis. It has the purpose of squashing the first @p axis
     *                    dimensions together. For instance, given a [4x4x4x4] image,
     *                    when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, float beta = 1.0f, size_t axis = 1);

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
     * @param[in] input  Original source tensor.
     * @param[in] output Original destination tensor.
     * @param[in] axis   (Optional) Reduction axis. It has the purpose of squashing the first @p axis
     *                    dimensions together. For instance, given a [4x4x4x4] image,
     *                    when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     */
    void configure_reshape_input_kernel(const ICLTensor *input, const ICLTensor *output, size_t axis);
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
     * @param[in] axis            (Optional) Reduction axis. It has the purpose of squashing the first @p axis
     *                    dimensions together. For instance, given a [4x4x4x4] image,
     *                    when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     */
    void configure_reshape_input_kernel(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *output, size_t axis);

    MemoryGroup                    _memory_group;
    CLLogits1DMaxShiftExpSumKernel _max_shift_exp_sum_kernel;
    CLLogits1DNormKernel           _norm_kernel;
    std::unique_ptr<ICLKernel>     _flatten_kernel_ptr;
    CLReshapeLayerKernel           _reshape_kernel;
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
