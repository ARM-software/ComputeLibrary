/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;
class ITensorInfo;
class CLCompileContext;

/** Basic function to compute a SoftmaxLayer.
 *
 * Softmax is calculated by :
 * @f[ out = exp((x - max(x)) * beta) / sum(exp((x - max(x)) * beta)) @f]
 *
 * Log Softmax is calculated by :
 * @f[ out = (x - max(x) * beta) - log(\sum{e^{x - max(x) * beta}}) @f]
 *
 * This function runs the following operators/kernels:
 * -# If axis is not 0:
 * -# @ref opencl::ClPermute
 * -# @ref opencl::kernels::ClLogits1DNormKernel
 * -# @ref opencl::kernels::ClLogits1DMaxShiftExpSumKernel
 */
template <bool IS_LOG = false>
class CLSoftmaxLayerGeneric : public IFunction
{
public:
    /** Constructor */
    CLSoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Default destructor */
    ~CLSoftmaxLayerGeneric();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32 for Softmax and F16/F32 for Log Softmax
     * @param[out] output Destination tensor. Data types supported: same as @p input
     * @param[in]  beta   (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in]  axis   (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
     *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
     */
    void configure(const ICLTensor *input, ICLTensor *output, float beta = 1.0f, int32_t axis = 0);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32 for Softmax and F16/F32 for Log Softmax
     * @param[out] output          Destination tensor. Data types supported: same as @p input
     * @param[in]  beta            (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in]  axis            (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
     *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, float beta = 1.0f, int32_t axis = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSoftmaxLayer
     *
     * @param[in] input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32 for Softmax and F16/F32 for Log Softmax
     * @param[in] output Destination tensor. Data types supported: same as @p input
     * @param[in] beta   (Optional) A scaling factor for the exponent. Defaults to 1.f
     * @param[in] axis   (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
     *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, float beta = 1.0f, int32_t axis = 0);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

using CLSoftmaxLayer    = CLSoftmaxLayerGeneric<false>;
using CLLogSoftmaxLayer = CLSoftmaxLayerGeneric<true>;
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLSOFTMAXLAYER_H */
