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
#ifndef ARM_COMPUTE_NEPOOLINGLAYER_H
#define ARM_COMPUTE_NEPOOLINGLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to simulate a pooling layer with the specified pooling operation. This function calls the following kernels:
 *
 * -# @ref cpu::CpuPool2d
 */
class NEPoolingLayer : public IFunction
{
public:
    /** Constructor */
    NEPoolingLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPoolingLayer(const NEPoolingLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPoolingLayer &operator=(const NEPoolingLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEPoolingLayer(NEPoolingLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEPoolingLayer &operator=(NEPoolingLayer &&) = delete;
    /** Default destructor */
    ~NEPoolingLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @note F16 is supported for pool sizes 2 and 3 only
     * @note Source tensor is padded with -inf for MAX pooling and 0 otherwise
     *       Cases where pooling region is completely outside input tensor are only supported for floating point data type
     *
     * @param[in, out] input     Source tensor. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]     output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]      pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out]     indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info, ITensor *indices = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPoolingLayer
     *
     * @note F16 is supported for pool sizes 2 and 3 only
     *
     * @param[in] input     Source tensor info. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[in] indices   (optional) Tensor info of the indices of the maximal values. Data type supported: U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
#endif /* ARM_COMPUTE_NEPOOLINGLAYER_H */
