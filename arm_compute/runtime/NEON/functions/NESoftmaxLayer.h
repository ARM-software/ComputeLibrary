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
#ifndef ARM_COMPUTE_NESOFTMAXLAYER_H
#define ARM_COMPUTE_NESOFTMAXLAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to compute a SoftmaxLayer and a Log SoftmaxLayer. */
template <bool IS_LOG = false>
class NESoftmaxLayerGeneric : public IFunction
{
public:
    /** Constructor */
    NESoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESoftmaxLayerGeneric(const NESoftmaxLayerGeneric &) = delete;
    /** Default move constructor */
    NESoftmaxLayerGeneric(NESoftmaxLayerGeneric &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESoftmaxLayerGeneric &operator=(const NESoftmaxLayerGeneric &) = delete;
    /** Default move assignment operator */
    NESoftmaxLayerGeneric &operator=(NESoftmaxLayerGeneric &&);
    /** Default destructor */
    ~NESoftmaxLayerGeneric();
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
     * @param[in,out] input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32. If the width is not a
     *                       multiple of the internal processing block size, @ref NEFillBorder replicates the
     *                       last value of each row to the nearest multiple.
     * @param[out]    output Destination tensor. Data types supported: same as @p input.
     * @param[in]     beta   (Optional) A scaling factor for the exponent.
     * @param[in]     axis   (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
     *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
     */
    void configure(ITensor *input, ITensor *output, float beta = 1.0f, int32_t axis = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref NESoftmaxLayer
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output Destination tensor info. Data types supported: same as @p input
     * @param[in] beta   (Optional) A scaling factor for the exponent.
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

using NESoftmaxLayer    = NESoftmaxLayerGeneric<false>;
using NELogSoftmaxLayer = NESoftmaxLayerGeneric<true>;

} // namespace arm_compute
#endif /* ARM_COMPUTE_NESOFTMAXLAYER_H */
