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
#ifndef __ARM_COMPUTE_NESOFTMAXLAYER_H__
#define __ARM_COMPUTE_NESOFTMAXLAYER_H__

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEFlattenLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEReshapeLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Basic function to compute a SoftmaxLayer.
 *
 * Softmax is calculated by :
 * @f[ out = \frac{e^{x - max(x)}}{\sum{e^{x - max(x)}}} @f]
 *
 * This function runs the following kernels:
 * -# @ref NEFillBorderKernel
 * -# @ref NELogits1DMaxKernel
 * -# @ref NELogits1DSoftmaxKernel
 */
class NESoftmaxLayer : public IFunction
{
public:
    /** Constructor */
    NESoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESoftmaxLayer(const NESoftmaxLayer &) = delete;
    /** Default move constructor */
    NESoftmaxLayer(NESoftmaxLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESoftmaxLayer &operator=(const NESoftmaxLayer &) = delete;
    /** Default move assignment operator */
    NESoftmaxLayer &operator=(NESoftmaxLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in,out] input  Source tensor. Data types supported: QASYMM8/F16/F32. If the width is not a
     *                       multiple of the internal processing block size, @ref NEFillBorderKernel replicates the
     *                       last value of each row to the nearest multiple.
     * @param[out]    output Destination tensor. Data types supported: same as @p input.
     * @param[in]     beta   (Optional) A scaling factor for the exponent.
     * @param[in]     axis   (Optional) Reduction axis. Defaults to 1. Must be in range [1, input_num_dimensions).
     *                       It has the purpose of squashing the first @p axis dimensions together. For instance, given a [4x4x4x4] image,
     *                       when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     */
    void configure(ITensor *input, ITensor *output, float beta = 1.0f, size_t axis = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref NESoftmaxLayer
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/F16/F32.
     * @param[in] output Destination tensor info. Data types supported: same as @p input
     * @param[in] beta   (Optional) A scaling factor for the exponent.
     * @param[in] axis   (Optional) Reduction axis. Defaults to 1. Must be in range [1, input_num_dimensions).
     *                   It has the purpose of squashing the first @p axis dimensions together. For instance, given a [4x4x4x4] image,
     *                   when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     *
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
     * @param[in] axis   (Optional) Reduction axis. Defaults to 1. Must be in range [1, input_num_dimensions).
     *                   It has the purpose of squashing the first @p axis dimensions together. For instance, given a [4x4x4x4] image,
     *                   when @p axis is 2, the Softmax reduction will be applied on each of the [4x4] planes of the input image.
     */
    void configure_reshape_input_kernel(const ITensor *input, const ITensor *output, size_t axis);

    MemoryGroup                _memory_group;
    NELogits1DMaxKernel        _max_kernel;
    NELogits1DSoftmaxKernel    _softmax_kernel;
    std::unique_ptr<INEKernel> _flat_or_reshape_kernel_ptr;
    NEFillBorderKernel         _fill_border_kernel;
    NEReshapeLayerKernel       _reshape_kernel;
    Tensor                     _max;
    Tensor                     _tmp;
    Tensor                     _input_flattened;
    Tensor                     _output_flattened;
    bool                       _needs_flattening;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NESOFTMAXLAYER_H__ */
